import numpy as np
from tqdm import tqdm
import math
import pickle
import concurrent.futures
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import skimage.io as io
import random
import os
import matplotlib.patches as patches
from sklearn.metrics import classification_report
from collections import defaultdict


def compute_integral_image(img):
    width = img.shape[0]
    height = img.shape[1]
    ii = np.zeros((width + 1, height + 1))

    for i in range(width):
        for j in range(height):
            ii[i + 1, j + 1] = np.sum(img[:i + 1, :j + 1])
    return ii


def scan_image(img, clf, window_x, window_y, stride_x, stride_y, scale):
    res = []
    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
    x, y = img.shape
    for i in range(0, x - window_x + 1, stride_x):
        for j in range(0, y - window_y + 1, stride_y):
            temp = img[i: i + window_x, j: j + window_y]
            p = clf.classify(temp)

            if p == 1:
                res.append((i * scale, j * scale, (i + window_x)
                           * scale, (j + window_y) * scale))
    return res


def get_faces(img, clf):
    window_x = 19
    window_y = 19
    stride_x = 3
    stride_y = 3
    pred = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
    pred = pred + scan_image(gray, clf, window_x,
                             window_y, stride_x, stride_y, 10)

    #img = cv2.resize(img, (150, 150))
    #window_x = 150 // 6
    #window_y = 150 // 5
    # scan_image(img, clf, window_x, window_y, stride_x, stride_y, 2)
    return pred


class Feature:
    # region -> (x, y, width, height)
    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions

    def compute_feature(self, ii):
        result = 0

        if (len(self.positive_regions) != 0):
            for x, y, w, h in self.positive_regions:
                result += ii[x, y] + ii[x + w, y + h] - \
                    ii[x + w, y] - ii[x, y + h]

        if (len(self.negative_regions) != 0):
            for x, y, w, h in self.negative_regions:
                result -= ii[x, y] + ii[x + w, y + h] - \
                    ii[x + w, y] - ii[x, y + h]

        return result


class WeakClassifier:
    def __init__(self, feature, threshold, polarity):
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        return 1 if self.polarity * self.feature.compute_feature(x) < self.polarity * self.threshold else 0


class ViolaJones:
    def __init__(self, imgs, y, pos_num, neg_num):
        self.stages = []
        self.y = y
        self.pos_num = pos_num
        self.neg_num = neg_num

        self.weights = np.zeros(len(self.y))
        self.iis = []
        self.init_wights()
        self.features = self.build_features((19, 19))
        for x in range(len(self.y)):
            self.iis.append(compute_integral_image(imgs[x]))
        self.X = self.apply_features(self.features, self.iis)

    def init_wights(self):
        self.weights = np.zeros(len(self.y))
        for x in range(len(self.y)):

            if self.y[x] == 1:
                self.weights[x] = 1.0 / (2 * self.pos_num)
            else:
                self.weights[x] = 1.0 / (2 * self.neg_num)

    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        for w in tqdm(range(1, width+1)):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        # 2 rectangle features
                        immediate = (i, j, w, h)
                        right = (i + w, j, w, h)
                        if i + 2 * w < width:  # Horizontally Adjacent
                            features.append(Feature([right], [immediate]))
                        bottom = (i, j+h, w, h)
                        if j + 2 * h < height:  # Vertically Adjacent
                            features.append(Feature([immediate], [bottom]))
                        right_2 = (i + 2 * w, j, w, h)

                        # 3 rectangle features
                        if i + 3 * w < width:  # Horizontally Adjacent
                            features.append(
                                Feature([right], [right_2, immediate]))
                        bottom_2 = (i, j + 2 * h, w, h)
                        if j + 3 * h < height:  # Vertically Adjacent
                            features.append(
                                Feature([bottom], [bottom_2, immediate]))

                        # 4 rectangle features
                        bottom_right = (i + w, j + h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(
                                Feature([right, bottom], [immediate, bottom_right]))
                        j += 1
                    i += 1
        return features

    def apply_features(self, features, iis):
        X = np.zeros((len(features), len(iis)))
        i = 0
        for i in tqdm(range(len(features))):
            X[i] = list(map(features[i].compute_feature, iis))

        return X

    def train_weak(self, start, end):
        total_pos, total_neg = 0, 0
        for w, label in zip(self.weights, self.y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []

        for index, feature in enumerate(self.X[start: end]):

            applied_feature = sorted(
                zip(self.weights, feature, self.y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float(
                'inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights,
                            pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = self.features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = WeakClassifier(best_feature, best_threshold, best_polarity)
            classifiers.append(clf)

        best_clf = self.select_best(classifiers)[0]
        return best_clf

    def select_best(self, classifiers):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for i in range(len(self.y)):
                correctness = abs(clf.classify(self.iis[i]) - self.y[i])
                accuracy.append(correctness)
                error += self.weights[i] * correctness
            error = error / len(self.y)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def train_stage(self, t):
        l = len(self.X)
        clfs = []
        alphas = []
        for t in tqdm(range(t)):
            self.weights = self.weights / np.linalg.norm(self.weights)
            # -----
            results_clfs = []
            with concurrent.futures.ProcessPoolExecutor() as ex:
                results = [ex.submit(self.train_weak,
                                     int(l / 12) * i, int(l / 12) * (i + 1)) for i in range(10)]
                for f in concurrent.futures.as_completed(results):
                    results_clfs.append(f.result())

            # ------
            clf, error, accuracy = self.select_best(results_clfs)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                self.weights[i] = self.weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)

            alphas.append(alpha)
            clfs.append(clf)

        self.stages.append((clfs, alphas))

    def classify_stage(self, image, clfs, alphas, isIntegral=False):
        total = 0
        ii = None
        if isIntegral:
            ii = image
        else:
            ii = compute_integral_image(image)
        for alpha, clf in zip(alphas, clfs):
            total += alpha * clf.classify(ii)

        score = total / (0.5 * sum(alphas))
        return 1 if score >= 0.87 else 0

    def train(self, stages_count):
        for stage in stages_count:
            self.train_stage(stage)

    def classify(self, image, isIntegral=False):
        for stage in self.stages:
            stage_pred = self.classify_stage(
                image, stage[0], stage[1], isIntegral)
            if stage_pred == 0:
                return 0
        return 1

    def save(self, filename, save_train):
        if not(save_train):
            self.y = []
            self.weights = []
            self.iis = []
            self.features = []
            self.X = []

        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
