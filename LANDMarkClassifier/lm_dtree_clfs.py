from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample

from math import ceil

import numpy as np


class ETClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8, max_depth=5, max_trees=128):
        self.n_feat = n_feat
        self.max_depth = max_depth
        self.max_trees = max_trees

    def fit(self, X, y):
        if X.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X.shape[1])],
                size=ceil(X.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re, y_re = resample(X, y, n_samples=X.shape[0], stratify=y)

        X_re = X_re[:, self.features]

        self.classes_, y_counts = np.unique(y_re, return_counts=True)

        clf = ExtraTreesClassifier(
            n_estimators=self.max_trees, max_depth=self.max_depth
        )

        self.model_type = "nonlinear_etc"

        self.clf_model = clf.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):
        return self.clf_model.predict(X[:, self.features])

    def predict_proba(self, X):
        return self.clf_model.predict_proba(X[:, self.features])

    def decision_function(self, X):
        D = self.clf_model.predict_proba(X[:, self.features])

        return np.where(D > 0.5, 1, -1)
