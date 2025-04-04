from sklearn.utils import resample
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer

from math import ceil

import numpy as np

import torch as pyt
from torch.cuda import is_available as is_gpu_available

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

        if self.n_out > 2:
            self.O = pyt.nn.Softmax(dim=-1)

        else:
            self.O = pyt.nn.Sigmoid()

    def forward(self, x):
        o = self.IN(x)
        o = self.IN_Dr(o)
        o = self.D_1(o)
        o = self.A_1(o)
        o = self.Dr_1(o)
        o = self.D_2(o)
        o = self.A_2(o)
        o = self.Dr_2(o)
        logit = self.D_3(o)
        probs = self.O(logit)

        return logit, probs


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
        self.y_transformer = LabelBinarizer().fit(y)

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
        y_trf = self.y_transformer.transform(y_trf).astype(np.float32)

        # Determine if minimum class count exists
        self.classes_, y_counts = np.unique(y, return_counts=True)

        self.y_min = min(y_counts) * 0.8

        # Use neural network if more than 6 samples are present in the minority class
        if self.y_min > self.minority:
            self.n_in = X_trf.shape[1]

            self.n_out = self.classes_.shape[0]
            if self.n_out == 2:
                self.n_out = 1
            
            # Get device
            use_autocast = False
            if is_gpu_available():
                use_autocast = True
                device_type = "cuda:0"
                self.device = pyt.device("cuda:0")

            else:
                device_type = "cpu"
                self.device = pyt.device("cpu")

            # Prepare data
            X_trf = pyt.tensor(X_trf)
            y_trf = pyt.tensor(y_trf)

            dataset_train = pyt.utils.data.DataLoader(
                list(zip(X_trf, y_trf)),
                shuffle=True,
                batch_size=16
                )

            # Prepare model and load it onto the GPU or CPU
            self.model = LMNNet(n_in=self.n_in, n_out=self.n_out)
            self.model.to(self.device)

            # Prepare scheduler and optimizer
            optimizer=pyt.optim.AdamW(self.model.parameters(), lr = 0.01)

            # Prepare loss function
            if self.n_out > 2:
                loss_fn = pyt.nn.CrossEntropyLoss().to(self.device)

            else:
                loss_fn = pyt.nn.BCEWithLogitsLoss().to(self.device)

            scaler = pyt.amp.GradScaler(self.device)

            # Training loop
            for epoch in range(100):

                if is_gpu_available():
                    pyt.cuda.empty_cache()

                # Training steps
                self.model.train()

                for batch_num, batch in enumerate(dataset_train):
                    x_in, y_in = batch
                    x_in = x_in.to(self.device)
                    y_in = y_in.to(self.device)

                    with pyt.amp.autocast(
                        device_type=device_type,
                        dtype=pyt.bfloat16,
                        enabled=use_autocast
                    ):
                    
                        x_logit, x_probs = self.model(x_in)

                        # Calculate loss - BCE
                        total_loss = loss_fn(x_logit, y_in)
                    
                    # Backwards pass
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()

                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()

                self.params = self.model.state_dict()

                del self.model

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
            _, p = clf(X_tensor[start : start + 16])
            p = p.detach().cpu().numpy()
            predictions.extend(p)

        predictions = np.asarray(predictions)

        del clf

        return predictions

    def decision_function(self, X):
        D = self.predict_proba(X)

        D = np.where(D > 0.5, 1, -1)

        if self.n_out == 1:
            D = D.flatten()

        return D

    def predict(self, X):
        predictions = self.predict_proba(X)

        predictions = np.argmax(predictions, axis=1)

        predictions = np.asarray([self.y_transformer.classes_[x] for x in predictions])

        return predictions
