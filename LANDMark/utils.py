import numpy as np

##########################################################################################
# For Bagging Classifier
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.utils import resample
from joblib import Parallel, delayed, parallel_backend


def _parallel_build(estimator, X, y, max_samples_tree):

    if X.shape[0] <= max_samples_tree or max_samples_tree == -1:
        X_trf = X
        y_trf = y

    else:
        X_trf, y_trf = resample(
            X, y, replace=True, n_samples=max_samples_tree, stratify=y
        )

    trained_estimator = estimator.fit(X_trf, y_trf)

    return trained_estimator


class BaggingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, max_samples_tree, n_estimators, n_jobs):

        self.base_estimator = base_estimator
        self.max_samples_tree = max_samples_tree
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, X, y):

        self.classes_ = np.asarray(list(set(y)))
        self.classes_.sort()

        if self.n_jobs > 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_build)(
                    clone(self.base_estimator), X, y, self.max_samples_tree
                )
                for i in range(self.n_estimators)
            )

        else:
            self.estimators_ = [
                _parallel_build(clone(self.base_estimator), X, y, self.max_samples_tree)
                for _ in range(self.n_estimators)
            ]

        return self

    def predict(self, X):

        class_map = {entry: i for i, entry in enumerate(self.classes_)}

        if self.n_jobs > 1:
            prediction_results = Parallel(self.n_jobs)(
                delayed(self.estimators_[i].predict)(X)
                for i in range(self.n_estimators)
            )

        else:
            prediction_results = [
                self.estimators_[i].predict(X) for i in range(self.n_estimators)
            ]

        prediction_results = np.asarray(prediction_results)

        predictions = []

        for col in range(prediction_results.shape[1]):
            sample_pred = prediction_results[:, col]

            class_names, counts = np.unique(sample_pred, return_counts=True)

            if counts.shape[0] > 1:
                predictions.append(class_names[np.argmax(counts)])

            else:
                predictions.append(class_names[0])

        predictions = np.asarray(predictions)

        return predictions

    def predict_proba(self, X):

        class_map = {entry: i for i, entry in enumerate(self.classes_)}

        prediction_results = Parallel(self.n_jobs)(
            delayed(self.estimators_[i].predict)(X) for i in range(self.n_estimators)
        )

        prediction_results = np.asarray(prediction_results)

        prediction_probs = np.zeros(
            shape=(X.shape[0], len(self.classes_)), dtype=np.float
        )

        for col in range(prediction_results.shape[1]):
            sample_pred = prediction_results[:, col]

            class_names, counts = np.unique(sample_pred, return_counts=True)
            probs = counts / np.sum(counts)

            if probs.shape[0] == self.classes_.shape[0]:
                prediction_probs[col] = probs

            else:
                index = class_map[class_names[0]]
                prediction_probs[col, index] = 1.0

        return prediction_probs


##########################################################################################
# For LANDMark


def return_importance_scores(model):

    feature_importances = []

    for i, estimator in enumerate(model):
        feature_importances.append(estimator.feature_importances_)

    feature_importances = np.mean(feature_importances, axis=0)

    feature_sum = np.sum(feature_importances)
    if feature_sum > 0:
        feature_importances = feature_importances / feature_sum

    return feature_importances


##########################################################################################
import tensorflow as tf
import keras.backend as K


def get_centralized_gradients(optimizer, loss, params):
    """Compute the centralized gradients.

    From: https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow

    This function is ideally not meant to be used directly unless you are building a custom optimizer, in which case you
    could point `get_gradients` to this function. This is a modified version of
    `tf.keras.optimizers.Optimizer.get_gradients`.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.
        loss: Scalar tensor to minimize.
        params: List of variables.

    # Returns:
      A gradients tensor.

    # Reference:
        [Yong et al., 2020](https://arxiv.org/abs/2004.01461)
    """

    # We here just provide a modified get_gradients() function since we are trying to just compute the centralized
    # gradients at this stage which can be used in other optimizers.
    grads = []
    for grad in K.gradients(loss, params):
        grad_len = len(grad.shape)
        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError(
            "An operation has `None` for gradient. "
            "Please make sure that all of your ops have a "
            "gradient defined (i.e. are differentiable). "
            "Common ops without gradient: "
            "K.argmax, K.round, K.eval."
        )
    if hasattr(optimizer, "clipnorm") and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [
            tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads
        ]
    if hasattr(optimizer, "clipvalue") and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Create a centralized gradients functions for a specified optimizer.

    From: From: https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.

    # Usage:

    ```py
    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> opt.get_gradients = gctf.centralized_gradients_for_optimizer(opt)
    >>> model.compile(optimizer = opt, ...)
    ```
    """

    def get_centralized_gradients_for_optimizer(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_optimizer
