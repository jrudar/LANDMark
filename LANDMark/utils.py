import numpy as np

##########################################################################################
#For Bagging Classifier
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from joblib import Parallel, delayed, parallel_backend

def _parallel_build(estimator, X, y):

    trained_estimator = estimator.fit(X, y)

    return trained_estimator

class BaggingClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator, n_estimators, n_jobs):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
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

##########################################################################################
#For LANDMark and LANDMarkRFE
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import rankdata

def return_importance_scores(model):

    feature_importances = []
        
    for i, estimator in enumerate(model):
        feature_importances.append(estimator.feature_importances_)

    feature_importances = np.mean(feature_importances, axis = 0)

    feature_sum = np.sum(feature_importances)
    if feature_sum > 0:
        feature_importances = feature_importances / feature_sum

    return feature_importances

def stats(X, y, n_feat, n_groups, p_1, p_2, iteration):

    if iteration == 0:
        alpha = p_1

    else:
        alpha = p_2

    y_unique, y_counts = np.unique(y, return_counts = True)

    P = []
    for i in range(n_feat):
        for group_a in range(0, n_groups-1):
           
            comparison = X[:, :, group_a]

            a = np.where(y == y_unique[group_a], True, False)

            comp_a = comparison[a, i]

            multi_p = []
            for group_b in range(group_a + 1, n_groups):
                b = np.where(y == y_unique[group_b], True, False)

                comp_b = comparison[b, i]

                comb = np.hstack((comp_a, comp_b))

                if np.all(np.where(comb == comp_a[0], True, False)):
                    P.append(0.99)

                else:        
                    """
                    Use Welch t-test on ranks.
                    Ruxton GD. 2006. the unequal variance t-test is an underused alternative 
                    to Student’s t-test and the Mann–Whitney U test. Behavioral Ecology. 
                    doi:10.1093/beheco/ark016
                    """
                    y_stacked = np.hstack((y[a], y[b]))
                    y_ind = np.asarray([i for i in range(y_stacked.shape[0])])
                    ranked = rankdata(comb)

                    X_a = ranked[y_ind[np.where(y_stacked == y_unique[group_a], True, False)]]
                    X_b = ranked[y_ind[np.where(y_stacked == y_unique[group_b], True, False)]]

                    _, p_value = ttest_ind(X_a,
                                           X_b,
                                           axis = None,
                                           equal_var = False)

                    if n_groups == 2:
                        P.append(p_value)

                    else:
                        multi_p.append(p_value)
                
            if n_groups > 2:
                corrected_p = multipletests(multi_p, alpha, method = "bh")
                P.append(corrected_p[1].min())

    #Filtering
    if iteration < 2:
        retained_new = multipletests(P, 
                                     alpha = alpha, 
                                     method = "fdr_tsbky")[0]

    else:
        retained_new = multipletests(P, 
                                     alpha = p_2, 
                                     method = "fdr_tsbh")[0]

    return retained_new

##########################################################################################