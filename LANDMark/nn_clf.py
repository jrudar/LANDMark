import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" 

import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('shap').setLevel(logging.FATAL)

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale = "dynamic")
tf.keras.mixed_precision.experimental.set_policy(policy)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, GaussianNoise, Activation, AlphaDropout
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow_addons.activations import mish

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning

import shap as sh

from random import choice

from math import ceil

import warnings

warnings.filterwarnings("ignore", category = ConvergenceWarning)

from gc import collect

class RandomOracle(ClassifierMixin, BaseEstimator):

    def __init__(self, oracle = "Linear", n_feat = 0.8):

        self.oracle = "Linear"
        self.n_feat = n_feat

    def fit(self, X, y):

        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        if self.oracle == "Linear":
            #Select two points at random
            index = np.random.choice([i for i in range(X.shape[0])], size = 2, replace = False) 
            x = X[index]

            #Make sure two unique instances are chosen
            while np.array_equal(x[0, self.features], x[1, self.features]): 
                index = np.random.choice([i for i in range(X.shape[0])], size = 2, replace = False) 
                x = X[index]

            #Find the midpoint
            midpoint = np.sum(x[:, self.features], axis = 0) * 0.5 

            #Find the normal vector and intercept
            self.weights = x[1, self.features] - x[0, self.features] 
            self.intercept = 0 - np.dot(self.weights.T, midpoint)

        return self

    def decision_function(self, X):

        if self.oracle == "Linear":
            predictions = np.asarray([np.dot(self.weights.T, X[i, self.features]) + self.intercept for i in range(X.shape[0])])

        return predictions

    def get_imp_scores(self):

        return None

class LM2Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_feat = 0.8):

        self.n_feat = n_feat

    def fit(self, X, y):
    
        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        y_min = min(y_counts)

        if y_min > 6:
            clf_1 = LogisticRegressionCV(max_iter = 2000, cv = 5)

        else:
            clf_1 = LogisticRegression(max_iter = 2000)

        self.m_type = "denselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)
            
            scores = scores.sum(axis = 0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape = (X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

            scores = final_features

        return scores

class LM1Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self):

        pass

    def fit(self, X, y):
    
        X_re, y_re = resample(X, y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        y_min = min(y_counts)

        if y_min > 6:
            clf_1 = LogisticRegressionCV(max_iter = 2000, cv = 5, solver = "liblinear", penalty = "l1")

        else:
            clf_1 = LogisticRegression(max_iter = 2000, solver = "liblinear", penalty = "l1")

        self.m_type = "sparselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X)

    def decision_function(self, X):

        return self.clf_model.decision_function(X)

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)

            scores = scores.sum(axis = 0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

        return scores

class RMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_feat = 0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        clf_1 = RidgeClassifierCV(alphas = (0.001, 0.01, 0.1, 1.0, 10, 100, 1000))
        
        self.m_type = "denselinear"

        self.clf_model = clf_1.fit(X_re, y_re)

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)

            scores = scores.sum(axis = 0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape = (X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

            scores = final_features

        return scores
            
class SVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_feat = 0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        if min(y_counts) > 6:
            self.cv = GridSearchCV(LinearSVC(max_iter = 2000),
                                   param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10, 100]},
                                              cv = 5).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            self.clf_model = LinearSVC(max_iter = 2000).fit(X_re, y_re)
        
        self.m_type = "denselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)

            scores = scores.sum(axis = 0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            final_features = np.zeros(shape = (X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

            scores = final_features

        return scores

class SG1Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self):

        pass

    def fit(self, X, y):

        X_re, y_re = resample(X, y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        if min(y_counts) > 6:
            self.cv = GridSearchCV(SGDClassifier(max_iter = 2000, penalty = "elasticnet"),
                                   param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                                               "loss": ["hinge", "modified_huber"]},
                                               cv = 5).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            clf = choice([SGDClassifier(alpha = 1.0, loss = "hinge", max_iter = 2000, penalty = "elasticnet"),
                          SGDClassifier(alpha = 1.0, loss = "modified_huber", max_iter = 2000, penalty = "elasticnet")])

            self.clf_model = clf.fit(X_re, y_re)

        self.m_type = "sparselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X)

    def decision_function(self, X):

        return self.clf_model.decision_function(X)

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)

            scores = scores.sum(axis = 0)

            score_sum = scores.sum()
            if score_sum > 0:
                scores = scores / score_sum

            return scores

class SG2Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_feat = 0.8):

        self.n_feat = n_feat

    def fit(self, X, y):

        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_re, y_re = resample(X[:, self.features], y, n_samples = X.shape[0], stratify = y)

        self.classes_, y_counts = np.unique(y_re, return_counts = True)

        if min(y_counts) > 6:
            self.cv = GridSearchCV(SGDClassifier(max_iter = 2000),
                                   param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
                                               "loss": ["hinge", "modified_huber"]},
                                               cv = 5).fit(X_re, y_re)

            self.clf_model = self.cv.best_estimator_

        else:
            clf = choice([SGDClassifier(alpha = 1.0, loss = "hinge", max_iter = 2000),
                          SGDClassifier(alpha = 1.0, loss = "modified_huber", max_iter = 2000)])

            self.clf_model = clf.fit(X_re, y_re)

        self.m_type = "denselinear"

        return self, self.decision_function(X)

    def predict(self, X):

        return self.clf_model.predict(X[:, self.features])

    def decision_function(self, X):

        return self.clf_model.decision_function(X[:, self.features])

    def get_imp_scores(self, X):
        
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = self.clf_model.coef_

            scores = np.abs(scores)

            scores = np.sum(scores, axis = 0)

            feature_sum = scores.sum()
            if feature_sum > 0:
                scores = scores / feature_sum

            final_features = np.zeros(shape = (X.shape[1]))
            for i, loc in enumerate(self.features):
                final_features[loc] = scores[i]

            scores = final_features

        return scores

class ANNClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_feat = 0.8):
        
        self.n_feat = n_feat

    def fit(self, X, y):

        self.m_type = "nonlinear"

        if X.shape[1] >= 4:
            self.features = np.random.choice([i for i in range(X.shape[1])],
                                              size = ceil(X.shape[1] * self.n_feat),
                                              replace = False)

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_trf = X[:, self.features].astype(np.float32)

        X_t, X_v, y_t, y_v = train_test_split(X_trf, 
                                                y,
                                                train_size = 0.90,
                                                stratify = y)

        self.classes_, _ = np.unique(y, return_counts = True)

        y_transformer = OneHotEncoder().fit(y.reshape(-1,1))

        y_trf = y_transformer.transform(y_t.reshape(-1,1)).toarray().astype(np.int)
        y_v = y_transformer.transform(y_v.reshape(-1,1)).toarray().astype(np.int)

        def make_model():

            in_sz = 2

            IN = Input(X_trf.shape[1])

            RF_IN = RandomFourierFeatures(24, trainable = False, dtype = "float32")(IN)

            RF_D1 = Dense(24,
                         activation = mish)(RF_IN)

            RF_D2 = Dense(24,
                         activation = mish)(RF_D1)

            RF_D3 = Dense(16,
                         activation = mish)(RF_D2)

            D1 = Dense(256,
                       activation = mish)(IN)

            DR1 = AlphaDropout(0.2)(D1)

            D2 = Dense(128,
                        activation = mish)(DR1)

            DR2 = AlphaDropout(0.2)(D2)

            D3 = Dense(64,
                        activation = mish
                        )(DR2)

            DR3 = AlphaDropout(0.2)(D3)

            D4 = Dense(48,
                        activation = mish
                        )(DR3)

            D5 = Dense(24,
                        activation = mish
                        )(DR3)

            C = Concatenate()([D5, RF_D3])

            PRE_OUT = Dense(y_trf.shape[1])(C)

            OUT = Activation("softmax",
                            dtype = "float32",)(PRE_OUT)
            
            return IN, OUT

        early_stop = EarlyStopping(monitor = "val_loss",
                                   mode = "min",
                                   patience = 40, #Should be hyper-parameter
                                   min_delta = 0.0001) #was 0.0001. Should be hyper-parameter

        clf = make_model()

        model = Model(clf[0],
                      clf[1])

        model.compile(optimizer = Adam(),
                      loss = "categorical_crossentropy")
         
        train_data = tf.data.Dataset.from_tensor_slices((X_t.astype(np.float32), y_trf.astype(int)))
        train_data = train_data.shuffle(buffer_size = X_t.shape[0], reshuffle_each_iteration=True)
        train_data = train_data.batch(32).prefetch(2)

        valid_data = tf.data.Dataset.from_tensor_slices((X_v.astype(np.float32), y_v.astype(int))).batch(32)

        model.fit(train_data ,
                  epochs = 300, 
                  verbose = 0,
                  callbacks = early_stop,
                  validation_data = valid_data, 
                  shuffle = True)

        self.weights = model.get_weights()
        self.config = model.get_config()

        full_data = tf.data.Dataset.from_tensor_slices(X[:, self.features].astype(np.float32)).batch(32)
        D = model.predict(full_data)

        tf.keras.backend.clear_session()

        return self, D

    def predict_proba(self, X):

        clf_models = Model.from_config(self.config, 
                                       custom_objects = {"RandomFourierFeatures": RandomFourierFeatures})
        clf_models.set_weights(self.weights)

        predictions = clf_models.predict(X[:, self.features], batch_size = 32)

        tf.keras.backend.clear_session()

        return predictions

    def predict(self, X):

        clf_models = Model.from_config(self.config, 
                                       custom_objects = {"RandomFourierFeatures": RandomFourierFeatures})
        clf_models.set_weights(self.weights)

        predictions = clf_models.predict(X[:, self.features], batch_size = 32)

        predictions = np.argmax(predictions, axis = 1)
        
        predictions = np.asarray([self.classes_[entry] for entry in predictions])
           
        tf.keras.backend.clear_session()

        return predictions

    def get_imp_scores(self, X):

        clf_models = Model.from_config(self.config, 
                                       custom_objects = {"RandomFourierFeatures": RandomFourierFeatures})
        clf_models.set_weights(self.weights)

        sh.explainers._deep.deep_tf.op_handlers["Mish"] = sh.explainers._deep.deep_tf.nonlinearity_1d(0)
        sh.explainers._deep.deep_tf.op_handlers["mish"] = sh.explainers._deep.deep_tf.nonlinearity_1d(0)

        sh_exp = sh.GradientExplainer(clf_models,
                                      data = X[:, self.features])
        
        scores = np.abs(sh_exp.shap_values(X[:, self.features]))

        scores = np.sum(scores, axis = 0).mean(axis = 0)

        tf.keras.backend.clear_session()

        feature_sum = scores.sum()
        if feature_sum > 0:
            scores = scores / feature_sum

        final_features = np.zeros(shape = (X.shape[1]))
        for i, loc in enumerate(self.features):
            final_features[loc] = scores[i]

        scores = final_features

        collect()

        return scores
