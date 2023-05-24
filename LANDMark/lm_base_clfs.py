import logging
import os

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

import tensorflow as tf

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Activation
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow_addons.layers import NoisyDense
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import AdamW, Lookahead
from tensorflow_addons.losses import TripletSemiHardLoss

from .utils import centralized_gradients_for_optimizer

import numpy as np

import pandas as pd

from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import (
    RidgeClassifierCV,
    LogisticRegressionCV,
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import resample
from sklearn.ensemble import ExtraTreesClassifier

from random import choice

from math import ceil

from gc import collect


class RandomOracle(ClassifierMixin, BaseEstimator):
    def __init__(self, oracle="Linear", n_feat=0.8):
        self.oracle = "Linear"
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

        if self.oracle == "Linear":
            # Select two points at random
            index = np.random.choice(
                [i for i in range(X.shape[0])], size=2, replace=False
            )
            x = X[index]

            # Make sure two unique instances are chosen
            while np.array_equal(x[0, self.features], x[1, self.features]):
                index = np.random.choice(
                    [i for i in range(X.shape[0])], size=2, replace=False
                )
                x = X[index]

            # Find the midpoint
            midpoint = np.sum(x[:, self.features], axis=0) * 0.5

            # Find the normal vector and intercept
            self.weights = x[1, self.features] - x[0, self.features]
            self.intercept = 0 - np.dot(self.weights.T, midpoint)

        return self

    def decision_function(self, X):
        if self.oracle == "Linear":
            predictions = np.dot(X[:, self.features], self.weights.T) + self.intercept

        return predictions

    def predict(self, X):
        predictions = np.where(self.decision_function(X) > 0, 1, -1)

        return predictions


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

        clf_1 = ExtraTreesClassifier(
            n_estimators=self.max_trees, max_depth=self.max_depth
        )

        self.model_type = "nonlinear_etc"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):
        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):
        return self.clf_model.predict_proba(X[:, self.features])


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

        y_min = min(y_counts)

        if y_min > 6:
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
            if self.model_type == "lr_l2":
                self.clf = LogisticRegression(max_iter=2000).fit(X_re, y_re)

            elif self.model_type == "lr_l1":
                solver = "liblinear"
                if X.shape[0] >= 500:
                    solver = "saga"

                self.clf = LogisticRegression(
                    max_iter=2000, solver=solver, penalty="l1"
                ).fit(X_re, y_re)

            elif self.model_type == "sgd_l2":
                self.clf = choice(
                    [
                        SGDClassifier(alpha=1.0, loss="hinge", max_iter=2000),
                        SGDClassifier(alpha=1.0, loss="modified_huber", max_iter=2000),
                    ]
                )

                self.clf.fit(X_re, y_re)

            elif self.model_type == "sgd_l1":
                self.clf = choice(
                    [
                        SGDClassifier(
                            alpha=1.0, loss="hinge", max_iter=2000, penalty="elasticnet"
                        ),
                        SGDClassifier(
                            alpha=1.0,
                            loss="modified_huber",
                            max_iter=2000,
                            penalty="elasticnet",
                        ),
                    ]
                )

                self.clf.fit(X_re, y_re)

            elif self.model_type == "ridge":
                self.clf = RidgeClassifierCV(
                    alphas=(0.001, 0.01, 0.1, 1.0, 10, 100, 1000)
                ).fit(X_re, y_re)

            elif self.model_type == "lsvc":
                self.clf = LinearSVC(max_iter=2000).fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):
        return self.clf.predict(X[:, self.features])

    def decision_function(self, X):
        return self.clf.decision_function(X[:, self.features])


class ANNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):
        self.n_feat = n_feat

    def fit(self, X, y):
        self.model_type = "nonlinear_nnet"

        self.classes_, _ = np.unique(y, return_counts=True)

        y_transformer = OneHotEncoder().fit(y.reshape(-1, 1))

        if X.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X.shape[1])],
                size=ceil(X.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_trf = X[:, self.features].astype(np.float32)

        try:
            X_t, X_v, y_t, y_v = train_test_split(X_trf, y, train_size=0.90, stratify=y)

        except:
            X_idx = resample(
                np.asarray([i for i in range(X_trf.shape[0])]),
                replace=False,
                n_samples=int(0.90 * X.shape[0]),
                stratify=y,
            )

            X_t = X_trf[X_idx]
            y_t = y[X_idx]

            X_idx_v = np.asarray([i for i in range(X.shape[0]) if i not in X_idx])
            X_v = X_trf[X_idx_v]
            y_v = y[X_idx_v]

        y_trf = y_transformer.transform(y_t.reshape(-1, 1)).toarray().astype(np.int)
        y_v = y_transformer.transform(y_v.reshape(-1, 1)).toarray().astype(np.int)

        def make_model():
            IN = tf.keras.layers.Input(X_trf.shape[1])

            R = tf.keras.layers.experimental.RandomFourierFeatures(
                1024, trainable=False, dtype="float32"
            )(IN)

            R1 = tf.keras.layers.Dropout(0.5)(R)

            OUT = tf.keras.layers.Dense(y_trf.shape[1] * 16, activation=mish)(R1)

            OUT = tf.keras.layers.Dropout(0.25)(OUT)

            OUT = tf.keras.layers.Dense(y_trf.shape[1], activation=mish)(OUT)

            OUT = tf.keras.layers.Activation(activation="softmax", dtype="float32")(OUT)

            return IN, OUT

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=40,  # Should be hyper-parameter
            min_delta=0.001,
        )

        clf = make_model()

        model = Model(clf[0], clf[1])

        opt = Lookahead(AdamW(1e-4, beta_1=0.95))

        opt.get_gradients = centralized_gradients_for_optimizer(opt)

        model.compile(optimizer=opt, loss="categorical_crossentropy")

        train_data = tf.data.Dataset.from_tensor_slices(
            (X_t.astype(np.float32), y_trf.astype(int))
        )
        train_data = train_data.shuffle(
            buffer_size=X_t.shape[0], reshuffle_each_iteration=True
        )
        train_data = train_data.batch(32).prefetch(2)

        valid_data = tf.data.Dataset.from_tensor_slices(
            (X_v.astype(np.float32), y_v.astype(int))
        ).batch(32)

        model.fit(
            train_data,
            epochs=300,
            verbose=0,
            callbacks=early_stop,
            validation_data=valid_data,
            shuffle=True,
        )

        self.weights = model.get_weights()
        self.config = model.get_config()

        full_data = tf.data.Dataset.from_tensor_slices(
            X[:, self.features].astype(np.float32)
        ).batch(32)
        D = model.predict(full_data)

        tf.keras.backend.clear_session()

        return self, D

    def predict_proba(self, X):
        clf_models = Model.from_config(
            self.config, custom_objects={"RandomFourierFeatures": RandomFourierFeatures}
        )
        clf_models.set_weights(self.weights)

        predictions = clf_models.predict(X[:, self.features], batch_size=32)

        tf.keras.backend.clear_session()

        return predictions

    def predict(self, X):
        clf_models = Model.from_config(
            self.config, custom_objects={"RandomFourierFeatures": RandomFourierFeatures}
        )
        clf_models.set_weights(self.weights)

        predictions = clf_models.predict(X[:, self.features], batch_size=32)

        predictions = np.argmax(predictions, axis=1)

        predictions = np.asarray([self.classes_[entry] for entry in predictions])

        tf.keras.backend.clear_session()

        return predictions
