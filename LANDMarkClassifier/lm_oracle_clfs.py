import numpy as np

from math import ceil

from sklearn.base import ClassifierMixin, BaseEstimator

from scipy.sparse import issparse


class RandomOracle(ClassifierMixin, BaseEstimator):
    def __init__(self, oracle="Linear", n_feat=0.8):
        self.oracle = "Linear"
        self.n_feat = n_feat

    def fit(self, X, y):
        if issparse(X):
            X_not_sparse = X.toarray()

        else:
            X_not_sparse = X

        if X_not_sparse.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X_not_sparse.shape[1])],
                size=ceil(X_not_sparse.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X_not_sparse.shape[1])])

        if self.oracle == "Linear":
            # Select two points at random
            index = np.random.choice(
                [i for i in range(X_not_sparse.shape[0])], size=2, replace=False
            )
            x = X_not_sparse[index]

            # Make sure two unique instances are chosen
            while np.array_equal(x[0, self.features], x[1, self.features]):
                index = np.random.choice(
                    [i for i in range(X_not_sparse.shape[0])], size=2, replace=False
                )
                x = X_not_sparse[index]

            # Find the midpoint
            midpoint = np.sum(x[:, self.features], axis=0) * 0.5

            # Find the normal vector and intercept
            self.weights = x[1, self.features] - x[0, self.features]
            self.intercept = 0 - np.dot(self.weights.T, midpoint)

        return self

    def decision_function(self, X):
        if issparse(X):
            X_not_sparse = X.toarray()

        else:
            X_not_sparse = X

        if self.oracle == "Linear":
            predictions = (
                np.dot(X_not_sparse[:, self.features], self.weights.T) + self.intercept
            )

        return predictions

    def predict(self, X):
        predictions = np.where(self.decision_function(X) > 0, 1, -1)

        return predictions
