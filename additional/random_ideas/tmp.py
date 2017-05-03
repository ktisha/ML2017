import json

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def normalize_input(X):
    return (X.T / np.sum(X, axis=1)).T


def read_data_cuisine(path):
    with open(path) as train_f:
        train_data = json.loads(train_f.read())

    X_train = [x['ingredients'] for x in train_data]
    X_train = [dict(zip(x, np.ones(len(x)))) for x in X_train]

    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train).toarray()
    X_train = normalize_input(X_train)
    X_train = X_train.astype(np.float32)

    feature_names = np.array(vec.feature_names_)

    lbl = LabelEncoder()

    y_train = [y['cuisine'] for y in train_data]
    y_train = lbl.fit_transform(y_train).astype(np.int32)

    # label_names = lbl.classes_

    # print(label_names)
    print(feature_names.shape)
    return X_train, y_train


class NaiveBayes:
    def __init__(self, get_features):
        # counter for feature/category
        self.features = {}
        # document per category counter
        self.categories = {}

        self.get_features = get_features

    def increment_feature(self, feature, category):
        self.features.setdefault(feature, {})
        self.features[feature].setdefault(category, 0)
        self.features[feature][category] += 1

    def increment_category(self, category):
        self.categories.setdefault(category, 0)
        self.categories[category] += 1

    def count_features(self, feature, category):
        if feature in self.features and category in self.features[feature]:
            return float(self.features[feature][category])
        return 0.0

    def count_docs(self, category):
        if category in self.categories:
            return float(self.categories[category])
        return 0

    def total_docs(self):
        return sum(self.categories.values())

    def categories(self):
        return self.categories.keys()

    def train(self, item, category):
        # print(item)
        features = self.get_features(item)
        # increment counters for all feature in category
        for feature in features:
            self.increment_feature(feature, category)

        # increment category counter
        self.increment_category(category)

    def feature_probability(self, feature, cat):
        if self.count_docs(cat) == 0:
            return 0
        return self.count_features(feature, cat) / self.count_docs(cat)

    def classify(self, item, default=None):
        probs = {}
        max = 0.0
        for cat in self.categories:
            probs[cat] = 1
        for cat in self.categories:
            features = self.get_features(item)
            for feature in features:
                probability = self.feature_probability(feature, cat)
                if probability != 0:
                    probs[cat] *= probability

        for cat in self.categories:
            if probs[cat] > max:
                max = probs[cat]
                best = cat
            print(probs[cat])
        return best

    def getthreshold(self, best):
        return 0.01
