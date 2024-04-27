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
        prediction_results = [
            self.estimators_[i].predict(X) for i in range(self.n_estimators)
        ]

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
