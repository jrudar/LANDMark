import numpy as np

from scipy.stats import entropy

from math import ceil
from random import choice

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

from gc import collect

from .nn_clf import (
    LM1Classifier,
    LM2Classifier,
    RandomOracle,
    ANNClassifier,
    RMClassifier,
    SVClassifier,
    SG1Classifier,
    SG2Classifier,
    ETClassifier,
)


def purity_function(N, N_lab, L, R, y, purity_fun = "gain"):

    # Calculate Information Gain
    if purity_fun == "gain":
        
        L_outcome, L_counts = np.unique(y[L], return_counts=True)
        L_prob = L_counts / L_counts.sum()
        H_L = entropy(L_prob) * (L_counts.sum() / N)

        R_outcome, R_counts = np.unique(y[R], return_counts=True)
        R_prob = R_counts / R_counts.sum()
        H_R = entropy(R_prob) * (R_counts.sum() / N)

        H_parent = entropy(N_lab)

        IG = H_parent - H_L - H_R

        return IG


class PredictData:
    def __init__(self, node_lab):

        self.node_lab = node_lab

    def predict(self, X):

        predictions = np.asarray([self.node_lab for i in range(X.shape[0])])

        return predictions


class Node:
    def __init__(
        self,
        min_samples_in_leaf,
        max_depth,
        max_features,
        min_gain,
        impurity,
        use_lm_l2,
        use_lm_l1,
        use_nnet,
        nnet_min_samples,
        use_etc,
        etc_max_depth,
        etc_max_trees,
        N,
        current_depth,
        use_oracle,
    ):

        # Tree construction parameters
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_lm_l2 = use_lm_l2
        self.use_lm_l1 = use_lm_l1
        self.use_nnet = use_nnet
        self.nnet_min_samples = nnet_min_samples
        self.use_etc = use_etc
        self.etc_max_depth = etc_max_depth
        self.etc_max_trees = etc_max_trees
        self.N = N
        self.current_depth = current_depth
        self.use_oracle = use_oracle

        # Node parameters
        self.left = None
        self.right = None

        self.splitter = None
        self.features = None
        self.support = None
        self.gain = None
        self.feature_importances_ = None
        self.N_p = None
        self.split_type = None

        self.terminal = False
        self.label = None
        self.node_id = None

    # Select the best split point for a dataset
    def get_split(self, X, y):

        self.node_id = id(self)

        # Prepare the list of hyperplanes, gains, and model types
        hyperplane_list = []
        gains = []
        model_type = []

        # Determine the probability of a sample reaching this node
        N_p = X.shape[0] / self.N
        self.N_p = N_p

        # Determine the counts of each class at this node
        outcomes, counts_tr = np.unique(y, return_counts=True)
        counts_sum = counts_tr.sum()
        counts_prob = counts_tr / counts_sum
        counts_min = np.min(counts_tr)
        counts_min_prob = counts_min / counts_sum

        # If the node is pure, the maximim depth is reached, or the number of samples is too small, create a leaf
        if (
            entropy(counts_prob) == 0
            or counts_sum < self.min_samples_in_leaf
            or (self.current_depth >= self.max_depth and self.max_depth > -1)
            or counts_min <= 1
        ):

            leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

            self.label = leaf_predictions.predict
            self.terminal = True

            return self

        # Otherwise split
        else:
            # Determine if a random oracle is to be used
            if self.use_oracle == True:
                oracle_d = 1
                models_d = 2

            else:
                oracle_d = -1
                models_d = 1

            # Create a Random Oracle Splitter
            if self.current_depth == oracle_d:
                self.splitter = RandomOracle(n_feat=self.max_features).fit(X, y)

                D = self.splitter.decision_function(X)

                L = np.where(D > 0, True, False)
                R = np.where(D <= 0, True, False)

                IG = purity_function(self.N, counts_prob, L, R, y, self.impurity)

                self.gain = IG

                # Update the node
                self.feature_importances_ = self.splitter.get_imp_scores(X)
                self.feature_importances_ = (
                    self.feature_importances_ * self.N_p * self.gain
                )

                # Recursivly split
                self.left = Node(
                    self.min_samples_in_leaf,
                    self.max_depth,
                    self.max_features,
                    self.min_gain,
                    self.impurity,
                    self.use_lm_l2,
                    self.use_lm_l1,
                    self.use_nnet,
                    self.nnet_min_samples,
                    self.use_etc,
                    self.etc_max_depth,
                    self.etc_max_trees,
                    N=X.shape[0],
                    current_depth=self.current_depth + 1,
                    use_oracle=self.use_oracle,
                ).get_split(X[L], y[L])

                self.right = Node(
                    self.min_samples_in_leaf,
                    self.max_depth,
                    self.max_features,
                    self.min_gain,
                    self.impurity,
                    self.use_lm_l2,
                    self.use_lm_l1,
                    self.use_nnet,
                    self.nnet_min_samples,
                    self.use_etc,
                    self.etc_max_depth,
                    self.etc_max_trees,
                    N=X.shape[0],
                    current_depth=self.current_depth + 1,
                    use_oracle=self.use_oracle,
                ).get_split(X[R], y[R])

                return self

            # Split using a Linear or Neural Network Models
            if self.current_depth >= models_d:

                self.c_choice = choice([i for i in range(outcomes.shape[0])])

                # Train Linear Models - L2
                if self.use_lm_l2 == True:
                    for clf in [
                        LM2Classifier(n_feat=self.max_features),
                        RMClassifier(n_feat=self.max_features),
                        SVClassifier(n_feat=self.max_features),
                        SG2Classifier(n_feat=self.max_features),
                    ]:

                        model, D = clf.fit(X, y)

                        if D.ndim > 1:
                            D = D[:, self.c_choice]

                        L = np.where(D > 0, True, False)
                        R = np.where(D <= 0, True, False)

                        X_L_n = X[L].shape[0]
                        X_R_n = X[R].shape[0]

                        # Calculate Information Gain
                        if X_L_n > 0 and X_R_n > 0:

                            IG = purity_function(self.N, counts_prob, L, R, y, self.impurity)

                            gains.append(IG)
                            hyperplane_list.append((model, L, R))
                            model_type.append(model.m_type)

                # Train Linear Models - L1 / ElasticNet
                if self.use_lm_l1 == True:
                    for clf in [
                        LM1Classifier(n_feat=self.max_features),
                        SG1Classifier(n_feat=self.max_features),
                    ]:
                        model, D = clf.fit(X, y)

                        if D.ndim > 1:
                            D = D[:, self.c_choice]

                        L = np.where(D > 0, True, False)
                        R = np.where(D <= 0, True, False)

                        X_L_n = X[L].shape[0]
                        X_R_n = X[R].shape[0]

                        # Calculate Information Gain
                        if X_L_n > 0 and X_R_n > 0:

                            IG = purity_function(self.N, counts_prob, L, R, y, self.impurity)

                            gains.append(IG)
                            hyperplane_list.append((model, L, R))
                            model_type.append(model.m_type)

                # Train a Neural Network
                if self.use_nnet == True:
                    if X.shape[0] >= self.nnet_min_samples:
                        for clf in [ANNClassifier(n_feat=self.max_features)]:
                            model, D = clf.fit(X, y)

                            if D.ndim > 1:
                                D = D[:, self.c_choice]

                            L = np.where(D > 0.5, True, False)
                            R = np.where(D <= 0.5, True, False)

                            X_L_n = X[L].shape[0]
                            X_R_n = X[R].shape[0]

                            # Calculate Information Gain
                            if X_L_n > 0 and X_R_n > 0:

                                IG = purity_function(self.N, counts_prob, L, R, y, self.impurity)

                                gains.append(IG)
                                hyperplane_list.append((model, L, R))
                                model_type.append(model.m_type)

                # Train Decision Tree Models
                if self.use_etc == True:
                    for clf in [ETClassifier(n_feat=self.max_features,
                                             max_depth = self.etc_max_depth,
                                             max_trees = self.etc_max_trees)]:
                        model, D = clf.fit(X, y)

                        if D.ndim > 1:
                            D = D[:, self.c_choice]

                        L = np.where(D > 0.5, True, False)
                        R = np.where(D <= 0.5, True, False)

                        X_L_n = X[L].shape[0]
                        X_R_n = X[R].shape[0]

                        # Calculate Information Gain
                        if X_L_n > 0 and X_R_n > 0:

                            IG = purity_function(self.N, counts_prob, L, R, y, self.impurity)
                            
                            gains.append(IG)
                            hyperplane_list.append((model, L, R))
                            model_type.append(model.m_type)

                gains = np.asarray(gains)
                hyperplane_list = np.asarray(hyperplane_list, dtype="object")
                model_type = np.asarray(model_type, dtype=np.str)

                # Split the node
                if gains.shape[0] > 0:

                    # Filter models using gain
                    best_index = np.argmax(gains)
                    best_gain = gains[best_index]
                    best_gains = np.where(gains == best_gain, True, False)

                    hyperplane_list = hyperplane_list[best_gains]

                    best_hyperplane = choice(hyperplane_list)
                    L = best_hyperplane[1]
                    R = best_hyperplane[2]

                    self.gain = best_gain
                    self.splitter = best_hyperplane[0]

                    self.feature_importances_ = self.splitter.get_imp_scores(
                        X, self.c_choice
                    )
                    self.feature_importances_ = (
                        self.feature_importances_ * self.N_p * self.gain
                    )

                    # Recursivly split
                    self.left = Node(
                        self.min_samples_in_leaf,
                        self.max_depth,
                        self.max_features,
                        self.min_gain,
                        self.impurity,
                        self.use_lm_l2,
                        self.use_lm_l1,
                        self.use_nnet,
                        self.nnet_min_samples,
                        self.use_etc,
                        self.etc_max_depth,
                        self.etc_max_trees,
                        N=X.shape[0],
                        current_depth=self.current_depth + 1,
                        use_oracle=self.use_oracle,
                    ).get_split(X[L], y[L])

                    self.right = Node(
                        self.min_samples_in_leaf,
                        self.max_depth,
                        self.max_features,
                        self.min_gain,
                        self.impurity,
                        self.use_lm_l2,
                        self.use_lm_l1,
                        self.use_nnet,
                        self.nnet_min_samples,
                        self.use_etc,
                        self.etc_max_depth,
                        self.etc_max_trees,
                        N=X.shape[0],
                        current_depth=self.current_depth + 1,
                        use_oracle=self.use_oracle,
                    ).get_split(X[R], y[R])

                    return self

                # Create a Leaf
                else:
                    leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

                    self.label = leaf_predictions.predict
                    self.terminal = True

                    return self


class MTree(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        min_samples_in_leaf,
        max_depth,
        max_features,
        min_gain,
        impurity,
        use_oracle,
        bootstrap,
        use_lm_l2,
        use_lm_l1,
        use_nnet,
        nnet_min_samples,
        use_etc,
        etc_max_depth,
        etc_max_trees
    ):

        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_oracle = use_oracle
        self.bootstrap = bootstrap
        self.use_lm_l2 = use_lm_l2
        self.use_lm_l1 = use_lm_l1
        self.use_nnet = use_nnet
        self.nnet_min_samples = nnet_min_samples
        self.use_etc = use_etc
        self.etc_max_depth = etc_max_depth
        self.etc_max_trees = etc_max_trees

    def fit(self, X, y):

        self.classes_ = np.asarray(list(set(y)))
        self.classes_.sort()

        # Create the root node and begin splitting
        tree = Node(
            self.min_samples_in_leaf,
            self.max_depth,
            self.max_features,
            self.min_gain,
            self.impurity,
            self.use_lm_l2,
            self.use_lm_l1,
            self.use_nnet,
            self.nnet_min_samples,
            self.use_etc,
            self.etc_max_depth,
            self.etc_max_trees,
            N=X.shape[0],
            current_depth=1,
            use_oracle=self.use_oracle,
        )

        # Increase diversity by reampling with replacement
        if self.bootstrap:
            X_re, y_re = resample(X, y, replace=True, n_samples=X.shape[0], stratify=y)

        else:
            X_re = X
            y_re = y

        # Get the tree
        tree.get_split(X=X_re, y=y_re)

        self.LMTree = tree

        # Calculate Importances
        true_features = self._get_importances(self.LMTree)

        feature_importances = np.mean(true_features, axis=0)

        importance_sum = np.sum(feature_importances)
        if importance_sum > 0:
            self.feature_importances_ = feature_importances / importance_sum

        else:
            self.feature_importances_ = feature_importances_

        # Find the maximum depth
        self.max_depth = max(self._get_max_depth(self.LMTree))

        # Find all Node Ids
        self.all_ids = list(set(self._get_node_ids(self.LMTree)))

        collect()

        return self

    def _get_importances(self, node):

        feature_importances = []

        if node.terminal == False:
            if hasattr(node, "feature_importances_"):
                feature_importances.append(node.feature_importances_)

                f_imp_l = self._get_importances(node.left)
                f_imp_r = self._get_importances(node.right)

                feature_importances.extend(f_imp_l)
                feature_importances.extend(f_imp_r)

                return feature_importances

            else:
                f_imp_l = self._get_importances(node.left)
                f_imp_r = self._get_importances(node.right)

                feature_importances.extend(f_imp_l)
                feature_importances.extend(f_imp_r)

                return feature_importances

        else:
            return feature_importances

        return feature_importances

    def _get_max_depth(self, node):

        depth = []

        depth.extend([node.current_depth])

        if node.terminal == False:
            depth.extend(self._get_max_depth(node.left))
            depth.extend(self._get_max_depth(node.right))

        else:
            return depth

        return depth

    def _get_node_ids(self, node):

        ids = []

        if node.terminal == False:
            ids.extend(self._get_node_ids(node.left))
            ids.extend(self._get_node_ids(node.right))

        else:
            ids.append(node.node_id)

        return ids

    def _predict(self, X, current_node=None, root=True, sample_index=None):

        final_predictions = []

        true_index = None
        if root == True:
            true_index = np.asarray([i for i in range(X.shape[0])])

        else:
            true_index = sample_index

        node = None
        if not (current_node):
            node = self.LMTree

        else:
            node = current_node

        if node.terminal == False:

            if type(node.splitter) == ANNClassifier:
                D = node.splitter.predict_proba(X)

                if D.ndim > 1:
                    D = D[:, node.c_choice]

                L = np.where(D > 0.5, True, False)
                R = np.where(D <= 0.5, True, False)

            else:
                D = node.splitter.decision_function(X)

                if D.ndim > 1:
                    D = D[:, node.c_choice]

                L = np.where(D > 0, True, False)
                R = np.where(D <= 0, True, False)

            X_L = X[L]
            left = true_index[L]

            X_R = X[R]
            right = true_index[R]

            if left.shape[0] > 0:

                predictions_left = self._predict(X_L, node.left, False, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:

                predictions_right = self._predict(X_R, node.right, False, right)
                final_predictions.extend(predictions_right)

        elif node.terminal == True:

            predictions = node.label(X)
            predictions = np.asarray(
                [
                    (true_index[i], prediction)
                    for i, prediction in enumerate(predictions)
                ]
            )

            return predictions

        return final_predictions

    def predict(self, X):

        mapping = {class_name: i for i, class_name in enumerate(self.classes_)}

        tree_predictions = self._predict(X)

        tree_predictions = [(int(entry[0]), entry[1]) for entry in tree_predictions]
        tree_predictions.sort()

        collect()

        return np.asarray(tree_predictions)[:, 1]

    def _depth(self, X, current_node=None, root=True, sample_index=None):

        final_predictions = []

        true_index = None
        if root == True:
            true_index = np.asarray([i for i in range(X.shape[0])])

        else:
            true_index = sample_index

        node = None
        if not (current_node):
            node = self.LMTree

        else:
            node = current_node

        if node.terminal == False:

            left, right = node.splitter.splitter.predict_split(
                X, true_index
            )  # Get indicies which move left and right

            if left.shape[0] > 0:
                predictions_left = self._depth(X, node.left, False, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:
                predictions_right = self._depth(X, node.right, False, right)
                final_predictions.extend(predictions_right)

        elif node.terminal == True:

            return [(entry, node.current_depth) for entry in true_index]

        return final_predictions

    def depth(self, X):

        tree_predictions = self._depth(X)

        tree_predictions.sort()

        tree_predictions = [entry[1] for entry in tree_predictions]

        return tree_predictions

    def score(self, X, y):

        score = balanced_accuracy_score(y, self.predict(X))

        collect()

        return score

    def _proximity(self, X, current_node=None, root=True, sample_index=None):

        final_predictions = []

        true_index = None
        if root == True:
            true_index = np.asarray([i for i in range(X.shape[0])])

        else:
            true_index = sample_index

        node = None
        if not (current_node):
            node = self.LMTree

        else:
            node = current_node

        if node.terminal == False:

            if type(node.splitter) == ANNClassifier:
                D = node.splitter.predict_proba(X)

                if D.ndim > 1:
                    D = D[:, node.c_choice]

                L = np.where(D > 0.5, True, False)
                R = np.where(D <= 0.5, True, False)

            else:
                D = node.splitter.decision_function(X)

                if D.ndim > 1:
                    D = D[:, node.c_choice]

                L = np.where(D > 0, True, False)
                R = np.where(D <= 0, True, False)

            X_L = X[L]
            left = true_index[L]

            X_R = X[R]
            right = true_index[R]

            if left.shape[0] > 0:

                predictions_left = self._proximity(X_L, node.left, False, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:

                predictions_right = self._proximity(X_R, node.right, False, right)
                final_predictions.extend(predictions_right)

        elif node.terminal == True:

            return [(entry, node.node_id) for entry in true_index]

        return final_predictions

    def proximity(self, X):

        tree_predictions = self._proximity(X)

        tree_predictions.sort()

        col_dict = {col: i for i, col in enumerate(self.all_ids)}

        emb_matrix = np.zeros(shape=(X.shape[0], len(self.all_ids)), dtype=np.int)

        for entry in tree_predictions:
            row = entry[0]
            col = col_dict[entry[1]]

            emb_matrix[row, col] = 1

        return emb_matrix
