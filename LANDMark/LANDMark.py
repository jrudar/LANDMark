import numpy as np

#For fitting
from .utils import BaggingClassifier
from .score_fun import g_score

from .tree import MTree

#For class construction
from sklearn.base import ClassifierMixin, BaseEstimator
from abc import ABCMeta

from gc import collect

class LANDMarkClassifier(ClassifierMixin, BaseEstimator, metaclass = ABCMeta):
 
    def __init__(self, 
                 n_estimators = 64,
                 min_samples_in_leaf = 5, 
                 max_depth = -1,
                 max_features = 0.80,
                 min_gain = 0,
                 impurity = "gain",
                 use_oracle = True,
                 n_jobs = 4):
        
        #Tree construction parameters
        self.n_estimators = n_estimators
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_oracle = use_oracle

        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        y : Numpy array of class labels with shape (n_samples,)

        Returns
        -----------
        self : object, the fitted model
        """
        self.classes_ = np.asarray(list(set(y)))
        self.classes_.sort()

        self.n_classes_ = len(self.classes_)
           
        #Fit a model
        self.estimators_ = BaggingClassifier(base_estimator = MTree(min_samples_in_leaf = self.min_samples_in_leaf,
                                                                    max_depth = self.max_depth,
                                                                    max_features = self.max_features,
                                                                    min_gain = self.min_gain,
                                                                    impurity = self.impurity,
                                                                    use_oracle = self.use_oracle),
                                             n_estimators = self.n_estimators,
                                             n_samples = 1.0,
                                             n_jobs = self.n_jobs)

        self.estimators_.fit(X, y)

        self.avg_depth = np.mean([estimator.max_depth for estimator in self.estimators_.estimators_])

        #Calculate feature importances
        self.feature_importances_ = []
        for i, estimator in enumerate(self.estimators_.estimators_):
            self.feature_importances_.append(estimator.feature_importances_)

        self.feature_importances_ = np.mean(self.feature_importances_, axis = 0)

        feature_sum = np.sum(self.feature_importances_)
        if feature_sum > 0:
            self.feature_importances_ = self.feature_importances_ / feature_sum
        
        collect()

        return self

    def predict(self, X):
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
 
    def predict_proba(self, X):
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

    def score(self, X, y):

        score = g_score(y,
                        self.predict(X))

        collect()

        return score

    def depth(self, X):

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.depth(X))

        emb = np.vstack(tree_mats)

        collect()

        return emb

    def proximity(self, X):

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X))

        emb = np.hstack(tree_mats)

        collect()

        return emb