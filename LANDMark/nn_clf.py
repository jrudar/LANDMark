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

logging.getLogger("shap").setLevel(logging.ERROR)
tf.get_logger().setLevel("FATAL")

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Activation
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow_addons.layers import NoisyDense
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import AdamW, Lookahead

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

import shap as sh

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

    def get_imp_scores(self, X):

        explainer = sh.Explainer(
            (self.weights, self.intercept),
            masker=sh.maskers.Independent(X[:, self.features]),
        )

        scores = explainer(X[:, self.features], silent=True).values

        scores = np.abs(scores).mean(axis=0)

        score_sum = scores.sum()
        if score_sum > 0:
            scores = scores / score_sum

        final_features = np.zeros(shape=(X.shape[1]))
        for i, loc in enumerate(self.features):
            final_features[loc] = scores[i]

        return final_features


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

        y_min = min(y_counts)

        clf_1 = ExtraTreesClassifier(
            n_estimators=self.max_trees, max_depth=self.max_depth
        )

        self.m_type = "nonlinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.predict_proba(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(self.clf_model)

            raw_scores = explainer(X[:, self.features]).values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class LM2Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

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

        # X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts=True)

        y_min = min(y_counts)

        if y_min > 6:
            clf_1 = LogisticRegressionCV(max_iter=2000, cv=5)

        else:
            clf_1 = LogisticRegression(max_iter=2000)

        self.m_type = "denselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class LM1Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        X_re, y_re = resample(X, y, n_samples=X.shape[0], stratify=y)

        solver = "liblinear"
        if X.shape[0] >= 500:
            solver = "saga"

        if X.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X.shape[1])],
                size=ceil(X.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re = X_re[:, self.features]

        self.classes_, y_counts = np.unique(y_re, return_counts=True)

        y_min = min(y_counts)

        if y_min > 6:
            clf_1 = LogisticRegressionCV(
                max_iter=2000, cv=5, solver=solver, penalty="l1"
            )

        else:
            clf_1 = LogisticRegression(max_iter=2000, solver=solver, penalty="l1")

        self.m_type = "sparselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class RMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

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

        clf_1 = RidgeClassifierCV(alphas=(0.001, 0.01, 0.1, 1.0, 10, 100, 1000))

        self.m_type = "denselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class SVClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

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

        if min(y_counts) > 6:
            self.cv = GridSearchCV(
                LinearSVC(max_iter=2000),
                param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10, 100]},
                cv=5,
            ).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            self.clf_model = LinearSVC(max_iter=2000).fit(X_re, y_re)

        self.m_type = "denselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class SG1Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        X_re, y_re = resample(X, y, n_samples=X.shape[0], stratify=y)

        if X.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X.shape[1])],
                size=ceil(X.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re = X_re[:, self.features]

        self.classes_, y_counts = np.unique(y_re, return_counts=True)

        if min(y_counts) > 6:
            self.cv = GridSearchCV(
                SGDClassifier(max_iter=2000, penalty="elasticnet"),
                param_grid={
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                    "loss": ["hinge", "modified_huber"],
                },
                cv=5,
            ).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            clf = choice(
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

            self.clf_model = clf.fit(X_re, y_re)

        self.m_type = "sparselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class SG2Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

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

        # X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts=True)

        if min(y_counts) > 6:
            self.cv = GridSearchCV(
                SGDClassifier(max_iter=2000),
                param_grid={
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                    "loss": ["hinge", "modified_huber"],
                },
                cv=5,
            ).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            clf = choice(
                [
                    SGDClassifier(alpha=1.0, loss="hinge", max_iter=2000),
                    SGDClassifier(alpha=1.0, loss="modified_huber", max_iter=2000),
                ]
            )

            self.clf_model = clf.fit(X_re, y_re)

        self.m_type = "denselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X, hp=None):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            explainer = sh.Explainer(
                self.clf_model, sh.maskers.Independent(X[:, self.features])
            )

            raw_scores = explainer(
                X[:, self.features], silent=True
            ).values  # .abs.values

            if self.classes_.shape[0] > 2:
                scores = np.abs(raw_scores[:, :, hp]).mean(axis=0)

            else:
                scores = np.abs(raw_scores).mean(axis=0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape=(X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

        return final_features


class ANNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        self.m_type = "nonlinear"

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

    def get_imp_scores(self, X, hp=None):

        clf_models = Model.from_config(
            self.config, custom_objects={"RandomFourierFeatures": RandomFourierFeatures}
        )
        clf_models.set_weights(self.weights)

        sh.explainers._deep.deep_tf.op_handlers[
            "Mish"
        ] = sh.explainers._deep.deep_tf.nonlinearity_1d(0)
        sh.explainers._deep.deep_tf.op_handlers[
            "mish"
        ] = sh.explainers._deep.deep_tf.nonlinearity_1d(0)

        sh_exp = sh.GradientExplainer(clf_models, data=X[:, self.features])

        scores = np.abs(sh_exp.shap_values(X[:, self.features]))

        if self.classes_.shape[0] > 2:
            scores = scores[hp]

        else:
            scores = scores[0]

        scores = scores.mean(axis=0)

        tf.keras.backend.clear_session()

        feature_sum = scores.sum()
        if feature_sum > 0:
            scores = scores / feature_sum

        final_features = np.zeros(shape=(X.shape[1]))
        for i, loc in enumerate(self.features):
            final_features[loc] = scores[i]

        scores = final_features

        collect()

        return scores