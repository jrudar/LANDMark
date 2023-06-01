from __future__ import annotations

import numpy as np

from .utils import Ensemble
from .tree import MTree

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from typing import Optional


class LANDMarkClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_estimators: int = 64,
        min_samples_in_leaf: int = 5,
        max_depth: Optional[int] = None,
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
        n_jobs: int = 4
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
        X_checked, y_checked = self._check_params(X, y)

        self.classes_ = np.unique(y_checked)

        # Fit a model
        self.estimators_ = Ensemble(
            base_estimator=MTree(
                min_samples_in_leaf=self.min_samples_in_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_gain=self.min_gain,
                impurity=self.impurity,
                q=self.q,
                use_oracle=self.use_oracle,
                use_lm_l2=self.use_lm_l2,
                use_lm_l1=self.use_lm_l1,
                use_nnet=self.use_nnet,
                nnet_min_samples=self.nnet_min_samples,
                use_etc=self.use_etc,
                etc_max_depth=self.etc_max_depth,
                etc_max_trees=self.etc_max_trees,
                resampler=self.resampler,
            ),
            n_estimators=self.n_estimators,
            class_names=self.classes_,
            n_jobs=self.n_jobs,
        )

        self.estimators_.fit(X_checked, y_checked)

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
        check_is_fitted(self, attributes=["classes_", "estimators_"])

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
        check_is_fitted(self, attributes=["classes_", "estimators_"])

        predictions = self.estimators_.predict_proba(X)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        check_is_fitted(self, attributes=["classes_", "estimators_"])

        score = balanced_accuracy_score(y, self.predict(X))

        return score

    def proximity(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["classes_", "estimators_"])

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X))

        emb = np.hstack(tree_mats)

        return emb

    def _check_params(self, X: np.ndarray, y: np.ndarray):
        SUPPORTED_IMPURITY = {"gain", "gain-ratio", "tsallis", "tsallis-gain-ratio"}

        # Check that X and y meet the minimum requirements
        X_conv, y_conv = check_X_y(X, y, accept_sparse=False)

        if not isinstance(self.n_estimators, int):
            raise TypeError("'n_estimators' must be an integer.")

        if isinstance(self.n_estimators, int):
            if self.n_estimators <= 0:
                raise ValueError("'n_estimators' must be greater than zero.")

        if not isinstance(self.min_samples_in_leaf, int):
            raise TypeError("'min_samples_in_leaf' must be an integer.")

        if isinstance(self.min_samples_in_leaf, int):
            if self.min_samples_in_leaf <= 0:
                raise ValueError("'min_samples_in_leaf' must be greater than zero.")

        if isinstance(self.max_depth, type(None)):
            pass
        elif isinstance(self.max_depth, int):
            if self.max_depth <= 0:
                raise ValueError("'max_depth' must be an greater than zero.")
        else:
            raise TypeError("'max_depth' must be an integer greater than zero or None.")

        if not isinstance(self.max_features, float):
            raise TypeError("'max_features' must be float.")

        if isinstance(self.max_features, float):
            if self.max_features <= 0 or self.max_features > 1:
                raise ValueError(
                    "'max_features' must be greater than zero but less than or equal to one."
                )

        if isinstance(self.min_gain, float):
            if self.min_gain < 0:
                raise ValueError("'min_gain' must be greater than or equal to zero.")
        else:
            raise TypeError("'min_gain' must be float.")

        if isinstance(self.impurity, str):
            if self.impurity not in SUPPORTED_IMPURITY:
                raise ValueError(
                    "Supplied 'impurity' is not supported. 'impurity' must be 'gain', 'gain-ratio', 'tsallis', or 'tsallis-gain-ratio'."
                )

        else:
            raise TypeError(
                "'impurity' must be 'gain', 'gain-ratio', 'tsallis', or 'tsallis-gain-ratio'."
            )

        if not isinstance(self.q, float):
            raise TypeError("'q' must be float.")

        if not isinstance(self.use_oracle, bool):
            raise TypeError("'use_oracle' must be True or False.")

        if not isinstance(self.use_lm_l2, bool):
            raise TypeError("'use_lm_l2' must be True or False.")

        if not isinstance(self.use_lm_l1, bool):
            raise TypeError("'use_lm_l1' must be True or False.")

        if not isinstance(self.use_nnet, bool):
            raise TypeError("'use_nnet' must be True or False.")

        if not isinstance(self.nnet_min_samples, int):
            raise TypeError("'nnet_min_samples' must be an integer.")

        if isinstance(self.nnet_min_samples, int):
            if self.nnet_min_samples <= 0:
                raise ValueError("'nnet_min_samples' must be greater than zero.")

        if not isinstance(self.use_etc, bool):
            raise TypeError("'use_etc' must be True or False.")

        if not isinstance(self.etc_max_depth, int):
            raise TypeError("'etc_max_depth' must be an integer.")

        if isinstance(self.etc_max_depth, int):
            if self.etc_max_depth <= 0:
                raise ValueError("'etc_max_depth' must be greater than zero.")

        if not isinstance(self.etc_max_trees, int):
            raise TypeError("'etc_max_trees' must be an integer.")

        if isinstance(self.etc_max_trees, int):
            if self.etc_max_trees <= 0:
                raise ValueError("'etc_max_trees' must be greater than zero.")

        if not isinstance(self.n_jobs, int):
            raise TypeError("'n_jobs' must be an integer.")

        if isinstance(self.n_jobs, int):
            if self.n_jobs <= 0:
                raise ValueError("'n_jobs' must be greater than zero.")

        if isinstance(self.resampler, type(None)):
            pass

        elif hasattr(self.resampler, "fit_transform") == False:
            raise ValueError("'resampler' must have a 'fit_transform(X, y)' function.")

        return X_conv, y_conv
