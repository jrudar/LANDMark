import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, clone

from joblib import Parallel, delayed, parallel_backend

def _parallel_build(estimator, X, y):

    trained_estimator = estimator.fit(X, y)

    return trained_estimator

class BaggingClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator, n_estimators, n_samples, n_jobs):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.n_jobs = n_jobs

    def fit(self, X, y):

        self.classes_ = np.asarray(list(set(y)))
        self.classes_.sort()
            
        self.estimators_ = Parallel(n_jobs = self.n_jobs)(delayed(_parallel_build)(clone(self.base_estimator), 
                                                                                   X, 
                                                                                   y)
                                                          for i in range(self.n_estimators))

        return self

    def predict(self, X):
    
        class_map = {entry: i for i, entry in enumerate(self.classes_)}

        prediction_results = Parallel(self.n_jobs)(delayed(self.estimators_[i].predict)(X)
                                                   for i in range(self.n_estimators))

        prediction_results = np.asarray(prediction_results)

        predictions = []

        for col in range(prediction_results.shape[1]):
            sample_pred = prediction_results[:, col]

            class_names, counts = np.unique(sample_pred, return_counts = True)

            if counts.shape[0] > 1:
                predictions.append(class_names[np.argmax(counts)])

            else:
                predictions.append(class_names[0])

        predictions = np.asarray(predictions)

        return predictions

    def predict_proba(self, X):

        class_map = {entry: i for i, entry in enumerate(self.classes_)}

        prediction_results = Parallel(self.n_jobs)(delayed(self.estimators_[i].predict)(X)
                                                   for i in range(self.n_estimators))

        prediction_results = np.asarray(prediction_results)

        prediction_probs = np.zeros(shape = (X.shape[0], len(self.classes_)),
                                    dtype = np.float)

        for col in range(prediction_results.shape[1]):
            sample_pred = prediction_results[:, col]

            class_names, counts = np.unique(sample_pred, return_counts = True)
            probs = counts / np.sum(counts)

            if probs.shape[0] == self.classes_.shape[0]:
                prediction_probs[col] = probs

            else:
                index = class_map[class_names[0]]
                prediction_probs[col, index] = 1.0

        return prediction_probs