import logging
import os

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import (
    RidgeClassifierCV,
    LogisticRegressionCV,
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from random import choice

from math import ceil


class LMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model_type, n_feat=0.8):
        self.model_type = model_type
        self.n_feat = n_feat

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

        self.y_min = min(y_counts)
        
        if self.y_min > 6:
            if self.model_type == "lr_l2":
                self.clf = LogisticRegressionCV(max_iter=2000, cv=5).fit(X_re, y_re)

            elif self.model_type == "lr_l1":
                solver = "liblinear"
                if X.shape[0] >= 500:
                    solver = "saga"

                self.clf = LogisticRegressionCV(
                    max_iter=2000, cv=5, solver=solver, penalty="l1"
                ).fit(X_re, y_re)

            elif self.model_type == "sgd_l2":
                self.cv = GridSearchCV(
                    SGDClassifier(max_iter=2000),
                    param_grid={
                        "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                        "loss": ["hinge", "modified_huber"],
                    },
                    cv=5,
                ).fit(X_re, y_re)

                self.clf = self.cv.best_estimator_

            elif self.model_type == "sgd_l1":
                self.cv = GridSearchCV(
                    SGDClassifier(max_iter=2000, penalty="elasticnet"),
                    param_grid={
                        "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                        "loss": ["hinge", "modified_huber"],
                    },
                    cv=5,
                ).fit(X_re, y_re)

                self.clf = self.cv.best_estimator_

            elif self.model_type == "ridge":
                self.clf = RidgeClassifierCV(
                    alphas=(0.001, 0.01, 0.1, 1.0, 10, 100, 1000), cv=5
                ).fit(X_re, y_re)

            elif self.model_type == "lsvc":
                self.cv = GridSearchCV(
                    LinearSVC(max_iter=2000),
                    param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10, 100]},
                    cv=5,
                ).fit(X_re, y_re)

                self.clf = self.cv.best_estimator_

        else:
            self.clf = ExtraTreesClassifier(n_estimators = 128, max_depth = 1)
            
            self.clf.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):
        return self.clf.predict(X[:, self.features])

    def decision_function(self, X):
        
        if self.y_min > 6:
            return self.clf.decision_function(X[:, self.features])

        else:
            return self.clf.predict_proba(X[:, self.features])