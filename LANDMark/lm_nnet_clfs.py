from sklearn.utils import resample
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

from math import ceil

import numpy as np

import torch as pyt

from skorch import NeuralNetClassifier

class LMNNet(pyt.nn.Module):
    def __init__(self, n_in, n_out):
        super(LMNNet, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.IN = pyt.nn.Linear(in_features = self.n_in, out_features = self.n_out * 32)
        self.IN_Dr = pyt.nn.Dropout(0.5)
        
        self.D_1 = pyt.nn.Linear(in_features = self.n_out * 32, out_features = self.n_out * 16)
        self.A_1 = pyt.nn.Mish()
        self.Dr_1 = pyt.nn.Dropout(0.375)

        self.D_2 = pyt.nn.Linear(in_features = self.n_out * 16, out_features = self.n_out * 8)
        self.A_2 = pyt.nn.Mish()
        self.Dr_2 = pyt.nn.Dropout(0.375)

        self.D_3 = pyt.nn.Linear(in_features = self.n_out * 8, out_features = self.n_out)
        self.O = pyt.nn.Softmax(dim = -1)

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
    def __init__(self, n_feat=0.8):
        self.n_feat = n_feat

    def fit(self, X, y):
        self.model_type = "nonlinear_nnet"

        self.classes_, _ = np.unique(y, return_counts=True)

        self.y_transformer = LabelEncoder().fit(y)

        if X.shape[1] >= 4:
            self.features = np.random.choice(
                [i for i in range(X.shape[1])],
                size=ceil(X.shape[1] * self.n_feat),
                replace=False,
            )

        else:
            self.features = np.asarray([i for i in range(X.shape[1])])

        X_trf, y_trf = resample(X[:, self.features], y, n_samples=X.shape[0], stratify=y)
        X_trf = X_trf.astype(np.float32)
        y_trf = self.y_transformer.transform(y_trf).astype(np.int64)

        self.n_in = X_trf.shape[1]
        self.n_out = self.classes_.shape[0]

        if pyt.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        clf = NeuralNetClassifier(LMNNet(n_in = X_trf.shape[1], n_out = self.classes_.shape[0]),
                             optimizer = pyt.optim.AdamW,
                             lr = 0.001,
                             max_epochs = 100,
                             batch_size = 16,
                             device = device,
                             iterator_train__shuffle=True,
                             verbose = 0
                             )

        clf.fit(X_trf, y_trf)
        
        self.params = clf.module.state_dict()

        with pyt.inference_mode():
            D = clf.predict_proba(X[:, self.features].astype(np.float32))

        return self, D

    def predict_proba(self, X):

        if pyt.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        clf = NeuralNetClassifier(LMNNet(n_in = self.n_in, n_out = self.n_out),
                                  optimizer = pyt.optim.AdamW,
                                  optimizer__lr = 0.001,
                                  max_epochs = 100,
                                  batch_size = 16,
                                  device = device
                                  )

        clf.module.load_state_dict(self.params)

        clf.initialize()

        with pyt.inference_mode():
            predictions = clf.predict_proba(X[:, self.features].astype(np.float32))

        return predictions

    def predict(self, X):

        predictions = self.predict_proba(X)

        predictions = np.argmax(predictions, axis = 1)

        return self.y_transformer.inverse_transform(predictions)

