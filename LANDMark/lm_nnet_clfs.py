from sklearn.utils import resample
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

from math import ceil

import numpy as np

import torch as pyt

from skorch import NeuralNetClassifier

from scipy.sparse import issparse


class LMNNet(pyt.nn.Module):
    def __init__(self, n_in, n_out):
        super(LMNNet, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.IN = pyt.nn.Linear(in_features=self.n_in, out_features=self.n_out * 32)
        self.A_1 = pyt.nn.Mish()
        self.IN_Dr = pyt.nn.Dropout(0.375)

        self.D_1 = pyt.nn.Linear(
            in_features=self.n_out * 32, out_features=self.n_out * 16
        )
        self.A_1 = pyt.nn.Mish()
        self.Dr_1 = pyt.nn.Dropout(0.375)

        self.D_2 = pyt.nn.Linear(
            in_features=self.n_out * 16, out_features=self.n_out * 16
        )  # n_out * 8
        self.A_2 = pyt.nn.Mish()
        self.Dr_2 = pyt.nn.Dropout(0.375)

        self.D_3 = pyt.nn.Linear(
            in_features=self.n_out * 16, out_features=self.n_out
        )  # in_features = *8
        self.O = pyt.nn.Softmax(dim=-1)

    def forward(self, x):
        o = self.IN(x)
        o = self.IN_Dr(o)
        o = self.D_1(o)
        o = self.A_1(o)
        o = self.Dr_1(o)
        o = self.D_2(o)
        o = self.A_2(o)
        o = self.Dr_2(o)
        o = self.D_3(o)
        o = self.O(o)

        return o


class ANNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_feat=0.8, minority=6, use_etc_split=True):
        self.n_feat = n_feat
        self.minority = minority

    def fit(self, X, y):
        self.model_type = "nonlinear_nnet"

        if issparse(X):
            X_not_sparse = X.toarray()

        else:
            X_not_sparse = X

        # Encode y
        self.y_transformer = LabelEncoder().fit(y)

        # Select features
        if X_not_sparse.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X_not_sparse.shape[1])],
                size=ceil(X_not_sparse.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X_not_sparse.shape[1])])

        # Bootstrap resample
        X_trf, y_trf = resample(
            X_not_sparse[:, self.features],
            y,
            n_samples=X_not_sparse.shape[0],
            stratify=y,
        )
        X_trf = X_trf.astype(np.float32)
        y_trf = self.y_transformer.transform(y_trf).astype(np.int64)

        # Determine if minimum class count exists
        self.classes_, y_counts = np.unique(y_trf, return_counts=True)

        self.y_min = min(y_counts) * 0.8

        # Use neural network if more than 6 samples are present in the minority class
        if self.y_min > self.minority:
            self.n_in = X_trf.shape[1]
            self.n_out = self.classes_.shape[0]

            if pyt.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            clf = NeuralNetClassifier(
                LMNNet(n_in=X_trf.shape[1], n_out=self.classes_.shape[0]),
                optimizer=pyt.optim.AdamW,
                lr=0.001,
                max_epochs=100,
                batch_size=16,
                device=device,
                iterator_train__shuffle=True,
                verbose=0,
            )

            clf.fit(X_trf, y_trf)

            self.params = clf.module.state_dict()

            del clf

            return self, self.decision_function(X)

        # Otherwise use an Extra Trees Classifier or Nothing
        else:
            return self, None

    def predict_proba(self, X):
        if issparse(X):
            X_not_sparse = X.toarray()

        else:
            X_not_sparse = X

        clf = LMNNet(n_in=self.n_in, n_out=self.n_out)

        clf.load_state_dict(self.params)

        n_batch = pyt.arange(0, len(X_not_sparse), 16)

        X_tensor = pyt.tensor(X_not_sparse[:, self.features].astype(np.float32))

        predictions = []
        for start in n_batch:
            p = clf(X_tensor[start : start + 16]).detach().cpu().numpy()
            predictions.extend(p)

        predictions = np.asarray(predictions)

        del clf

        return predictions

    def decision_function(self, X):
        D = self.predict_proba(X)

        return np.where(D > 0.5, 1, -1)

    def predict(self, X):
        predictions = self.predict_proba(X)

        predictions = np.argmax(predictions, axis=1)

        return self.y_transformer.inverse_transform(predictions)
