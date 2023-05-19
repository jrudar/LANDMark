from __future__ import annotations

import numpy as np

from .utils import Ensemble
from .tree import MTree

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import balanced_accuracy_score


class LANDMarkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 64,
        min_samples_in_leaf: int = 5,
        max_depth: int = -1,
        max_features: float = 0.80,
        min_gain: float = 0.0,
        impurity: str = "gain",
        q: float = 1.5,
        use_oracle: bool = True,
        use_lm_l2: bool = True,
        use_lm_l1: bool = True,
        use_nnet: bool = True,
        nnet_min_samples: int = 32,
        use_etc: bool = True,
        etc_max_depth: int = 5,
        etc_max_trees: int = 128,
        resampler = None,
        bootstrap: bool = False,
        n_jobs: int = 4,
    ):
        # Tree construction parameters
        self.n_estimators = n_estimators
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.q = q
        self.use_oracle = use_oracle
        self.use_lm_l2 = use_lm_l2
        self.use_lm_l1 = use_lm_l1
        self.use_nnet = use_nnet
        self.nnet_min_samples = nnet_min_samples
        self.use_etc = use_etc
        self.etc_max_depth = etc_max_depth
        self.etc_max_trees = etc_max_trees
        self.resampler = resampler
        self.bootstrap = bootstrap

        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> LANDMarkClassifier:
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        y : Numpy array of class labels with shape (n_samples,)

        Returns
        -----------
        self : object, the fitted model
        """
        self.classes_ = np.unique(y)

        # Fit a model
        self.estimators_ = Ensemble(
            base_estimator=MTree(
                min_samples_in_leaf=self.min_samples_in_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_gain=self.min_gain,
                impurity=self.impurity,
                q = self.q,
                use_oracle=self.use_oracle,
                bootstrap=self.bootstrap,
                use_lm_l2=self.use_lm_l2,
                use_lm_l1=self.use_lm_l1,
                use_nnet=self.use_nnet,
                nnet_min_samples=self.nnet_min_samples,
                use_etc=self.use_etc,
                etc_max_depth=self.etc_max_depth,
                etc_max_trees=self.etc_max_trees,
            ),
            resampler=self.resampler,
            n_estimators=self.n_estimators,
            class_names=self.classes_,
            n_jobs=self.n_jobs,
        )

        self.estimators_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of predictions with shape (n_samples,)
        """
        predictions = self.estimators_.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of probabilities with shape (n_samples, n_classes)
        """
        predictions = self.estimators_.predict_proba(X)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray)-> float:
        score = balanced_accuracy_score(y, self.predict(X))

        return score

    def proximity(self, X: np.ndarray) -> np.ndarray:
        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X))

        emb = np.hstack(tree_mats)

        return emb

    def _check_params(self, X: np.ndarray, y: np.ndarray):
        pass
