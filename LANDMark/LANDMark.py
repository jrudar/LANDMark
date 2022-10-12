import numpy as np

# For fitting
from .utils import BaggingClassifier, return_importance_scores
from .tree import MTree

# For class construction
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import balanced_accuracy_score

from abc import ABCMeta, abstractmethod
from gc import collect


class LANDMarkModel(BaseEstimator, metaclass=ABCMeta):
    """Base Class for all LANDMark Classification Models"""

    def fit(self, X, y):
        """Fits a LANDMark Model"""

        return self._fit(X, y)

    def predict(self, X):
        """Predict Using a LANDMark Model"""

        return self._predict(X)

    def predict_proba(self, X):
        """Return Class Probabilities Using a LANDMark Model"""

        return self._predict_proba(X)

    def score(self, X, y):
        """Returns the Balanced Accuracy Score"""

        return self._score(X)

    def proximity(self, X):
        """Returns the Tree Embedding"""

        return self._proximity(X)

    @abstractmethod
    def _fit(self, X, y):
        """Fits a LANDMark Model"""

    @abstractmethod
    def _predict(self, X):
        """Predict Using a LANDMark Model"""

    @abstractmethod
    def _predict_proba(self, X):
        """Return Class Probabilities Using a LANDMark Model"""

    @abstractmethod
    def _score(self, X, y):
        """Returns the Balanced Accuracy Score"""

    @abstractmethod
    def _proximity(self):
        """Returns the Tree Embedding"""


class LANDMarkClassifier(ClassifierMixin, LANDMarkModel):
    def __init__(
        self,
        n_estimators=64,
        min_samples_in_leaf=5,
        max_depth=-1,
        max_features=0.80,
        min_gain=0,
        impurity="gain",
        use_oracle=True,
        use_lm_l2=True,
        use_lm_l1=True,
        use_nnet=True,
        nnet_min_samples=32,
        use_etc=True,
        etc_max_depth=5,
        etc_max_trees=128,
        max_samples_tree=-1,
        bootstrap=False,
        n_jobs=4,
    ):

        # Tree construction parameters
        self.n_estimators = n_estimators
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_oracle = use_oracle
        self.use_lm_l2 = use_lm_l2
        self.use_lm_l1 = use_lm_l1
        self.use_nnet = use_nnet
        self.nnet_min_samples = nnet_min_samples
        self.use_etc = use_etc
        self.etc_max_depth = etc_max_depth
        self.etc_max_trees = etc_max_trees
        self.max_samples_tree = max_samples_tree
        self.bootstrap = bootstrap

        self.n_jobs = n_jobs

    def _fit(self, X, y):
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

        self.n_classes_ = self.classes_.shape[0]

        # Fit a model
        self.estimators_ = BaggingClassifier(
            base_estimator=MTree(
                min_samples_in_leaf=self.min_samples_in_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_gain=self.min_gain,
                impurity=self.impurity,
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
            max_samples_tree=self.max_samples_tree,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
        )

        self.estimators_.fit(X, y)

        self.avg_depth = np.mean(
            [estimator.max_depth for estimator in self.estimators_.estimators_]
        )

        # Get feature importance scores
        self.feature_importances_ = return_importance_scores(
            self.estimators_.estimators_
        )

        collect()

        return self

    def _predict(self, X):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of predictions with shape (n_samples,)
        """
        predictions = self.estimators_.predict(X)

        collect()

        return predictions

    def _predict_proba(self, X):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of probabilities with shape (n_samples, n_classes)
        """
        predictions = self.estimators_.predict_proba(X)

        collect()

        return predictions

    def _score(self, X, y):

        score = balanced_accuracy_score(y, self.predict(X))

        collect()

        return score

    def _proximity(self, X):

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X))

        emb = np.hstack(tree_mats)

        collect()

        return emb
