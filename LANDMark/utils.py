import numpy as np

##########################################################################################
# For Bagging Classifier
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from scipy.special import softmax
from joblib import Parallel, delayed


def _parallel_build(estimator, X, y):
    return estimator.fit(X, y)


class Ensemble(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, n_estimators, class_names, n_jobs):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.classes_ = class_names
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_build)(clone(self.base_estimator), X, y)
            for i in range(self.n_estimators)
        )

        return self

    def predict(self, X):
        prediction_probs = self.predict_proba(X)

        max_probs = np.argmax(prediction_probs, axis=1)

        predictions = [self.classes_[idx] for idx in max_probs]

        predictions = np.asarray(predictions)

        return predictions

    def predict_proba(self, X):
        class_map = {class_name: i for i, class_name in enumerate(self.classes_)}

        # Returns an array that of shape (n_estimators, n_samples)
        prediction_results = Parallel(self.n_jobs)(
            delayed(self.estimators_[i].predict)(X) for i in range(self.n_estimators)
        )

        prediction_results = np.asarray(prediction_results)

        # Create an array that will hold all probabilities
        prediction_probs = np.zeros(
            shape=(X.shape[0], self.classes_.shape[0]), dtype=float
        )

        # Loop over samples and record the probabilities
        for sample_num in range(prediction_results.shape[1]):
            sample_pred = prediction_results[:, sample_num]

            class_names, counts = np.unique(sample_pred, return_counts=True)
            probs = counts / np.sum(counts)

            if probs.shape[0] == self.classes_.shape[0]:
                prediction_probs[sample_num] = probs

            elif probs.shape[0] == 1:
                index = class_map[class_names[0]]
                prediction_probs[sample_num, index] = 1.0

            else:
                for idx, prob in enumerate(probs):
                    index = class_map[class_names[idx]]
                    prediction_probs[sample_num, index] = prob

        # Ensure all probabilities sum to one
        prediction_probs = softmax(prediction_probs, axis=1)

        return prediction_probs


##########################################################################################
# For Neural Network Models
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
