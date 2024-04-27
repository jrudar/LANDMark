import numpy as np

from scipy.stats import entropy

from random import choice

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score

from .lm_linear_clfs import LMClassifier
from .lm_oracle_clfs import RandomOracle
from .lm_dtree_clfs import ETClassifier
from .lm_nnet_clfs import ANNClassifier


def tsallis_fun(N, N_lab, L, R, y, mode, q):
    if q == 1:  # Special case
        if "ratio" in mode.split("-"):
            return entropy_fun(N, N_lab, L, R, y, "gain-ratio")

        else:
            return entropy_fun(N, N_lab, L, R, y, "gain")

    scaler = 1 / (1 - q)

    L_outcome, L_counts = np.unique(y[L], return_counts=True)
    L_prob = L_counts / L_counts.sum()
    H_L = (L_counts.sum() / N) * (scaler * (np.power(L_prob, q).sum() - 1))

    R_outcome, R_counts = np.unique(y[R], return_counts=True)
    R_prob = R_counts / R_counts.sum()
    H_R = (R_counts.sum() / N) * (scaler * (np.power(R_prob, q).sum() - 1))

    H_parent = scaler * (np.power(N_lab, q).sum() - 1)

    IG = H_parent - H_R - H_L

    if mode == "tsallis":
        return IG

    else:
        norm_factor = np.asarray([(L_counts.sum() / N), (R_counts.sum() / N)])
        norm_factor = 1 + (scaler * (np.power(norm_factor, q).sum() - 1))

        GR = IG / norm_factor

        return GR


def entropy_fun(N, N_lab, L, R, y, mode):
    L_outcome, L_counts = np.unique(y[L], return_counts=True)
    L_prob = L_counts / L_counts.sum()
    H_L = entropy(L_prob) * (L_counts.sum() / N)

    R_outcome, R_counts = np.unique(y[R], return_counts=True)
    R_prob = R_counts / R_counts.sum()
    H_R = entropy(R_prob) * (R_counts.sum() / N)

    H_parent = entropy(N_lab)

    IG = H_parent - H_L - H_R

    if mode == "gain":
        return IG

    else:
        norm_factor = np.asarray([(L_counts.sum() / N), (R_counts.sum() / N)])
        norm_factor = 1 + entropy(norm_factor)

        GR = IG / norm_factor

        return GR


def purity_function(N, N_lab, L, R, y, purity_fun, q):
    if purity_fun == "gain" or purity_fun == "gain-ratio":
        return entropy_fun(N, N_lab, L, R, y, purity_fun)

    elif purity_fun == "tsallis-gain-ratio" or purity_fun == "tsallis":
        return tsallis_fun(N, N_lab, L, R, y, purity_fun, q)


class PredictData:
    def __init__(self, node_lab):
        self.node_lab = node_lab

    def predict(self, X):
        predictions = np.asarray([self.node_lab for i in range(X.shape[0])])

        return predictions


class Node:
    def __init__(self):
        # Node parameters
        self.c_choice = None

        self.left = None
        self.right = None

        self.splitter = None
        self.features = None
        self.gain = None
        self.split_type = None

        self.terminal = False
        self.label = None
        self.node_id = None

    # Select the best split point for a dataset
    def get_split(
        self,
        X,
        y,
        min_samples_in_leaf,
        max_depth,
        max_features,
        min_gain,
        impurity,
        q,
        use_lm_l2,
        use_lm_l1,
        minority_sz_lm,
        use_nnet,
        nnet_min_samples,
        minority_sz_nnet,
        use_etc,
        etc_max_depth,
        etc_max_trees,
        N,
        current_depth,
        use_oracle,
        use_cascade,
    ):
        # Get the ID of the node
        self.node_id = id(self)

        # Prepare the list of hyperplanes, gains, and model types
        hyperplane_list = []
        gains = []
        model_type = []

        # Determine the counts of each class at this node
        outcomes, counts_tr = np.unique(y, return_counts=True)
        counts_sum = counts_tr.sum()
        counts_prob = counts_tr / counts_sum
        counts_min = np.min(counts_tr)

        # Check if stopping criteria are met
        if entropy(counts_prob) == 0:
            leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

            self.label = leaf_predictions.predict
            self.terminal = True

            return self

        if counts_sum < min_samples_in_leaf:
            leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

            self.label = leaf_predictions.predict
            self.terminal = True

            return self

        if counts_min <= 1:
            leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

            self.label = leaf_predictions.predict
            self.terminal = True

            return self

        if not isinstance(max_depth, type(None)) and current_depth >= max_depth:
            leaf_predictions = PredictData(outcomes[np.argmax(counts_prob)])

            self.label = leaf_predictions.predict
            self.terminal = True

            return self

        # Otherwise split
        else:
            # Create a Random Oracle Splitter
            if use_oracle:
                self.splitter = RandomOracle(n_feat=max_features).fit(X, y)

                D = self.splitter.decision_function(X)

                # Extend X using the output of the decision function, D, if the cascade parameter is True
                if use_cascade:
                    X = np.hstack((X, D.reshape(-1, 1)))

                L = np.where(D > 0, True, False)
                R = np.where(D <= 0, True, False)

                IG = purity_function(counts_sum, counts_prob, L, R, y, impurity, q)

                self.gain = IG

                # Recursivly split
                self.left = Node().get_split(
                    X[L],
                    y[L],
                    min_samples_in_leaf=min_samples_in_leaf,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_gain=min_gain,
                    impurity=impurity,
                    q=q,
                    use_lm_l2=use_lm_l2,
                    use_lm_l1=use_lm_l1,
                    minority_sz_lm=minority_sz_lm,
                    use_nnet=use_nnet,
                    nnet_min_samples=nnet_min_samples,
                    minority_sz_nnet=minority_sz_nnet,
                    use_etc=use_etc,
                    etc_max_depth=etc_max_depth,
                    etc_max_trees=etc_max_trees,
                    N=X.shape[0],
                    current_depth=current_depth + 1,
                    use_oracle=False,
                    use_cascade=use_cascade,
                )

                self.right = Node().get_split(
                    X[R],
                    y[R],
                    min_samples_in_leaf=min_samples_in_leaf,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_gain=min_gain,
                    impurity=impurity,
                    q=q,
                    use_lm_l2=use_lm_l2,
                    use_lm_l1=use_lm_l1,
                    minority_sz_lm=minority_sz_lm,
                    use_nnet=use_nnet,
                    nnet_min_samples=nnet_min_samples,
                    minority_sz_nnet=minority_sz_nnet,
                    use_etc=use_etc,
                    etc_max_depth=etc_max_depth,
                    etc_max_trees=etc_max_trees,
                    N=X.shape[0],
                    current_depth=current_depth + 1,
                    use_oracle=False,
                    use_cascade=use_cascade,
                )

                return self

            # Split using a Linear, Tree, or Neural Network Models
            else:
                self.c_choice = choice([i for i in range(outcomes.shape[0])])

                # Train Linear Models - L2
                if use_lm_l2:
                    for clf in [
                        LMClassifier(
                            model_type="lr_l2",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                        LMClassifier(
                            model_type="sgd_l2",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                        LMClassifier(
                            model_type="ridge",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                        LMClassifier(
                            model_type="lsvc",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                    ]:
                        model, D = clf.fit(X, y)

                        if not isinstance(D, type(None)):
                            if D.ndim > 1:
                                D = D[:, self.c_choice]

                            L = np.where(D > 0, True, False)
                            R = np.where(D <= 0, True, False)

                            X_L_n = X[L].shape[0]
                            X_R_n = X[R].shape[0]

                            # Calculate Information Gain
                            if X_L_n > 0 and X_R_n > 0:
                                IG = purity_function(
                                    counts_sum, counts_prob, L, R, y, impurity, q
                                )

                                gains.append(IG)
                                hyperplane_list.append((model, L, R))
                                model_type.append(model.model_type)

                # Train Linear Models - L1 / ElasticNet
                if use_lm_l1:
                    for clf in [
                        LMClassifier(
                            model_type="lr_l1",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                        LMClassifier(
                            model_type="sgd_l1",
                            n_feat=max_features,
                            minority=minority_sz_lm,
                        ),
                    ]:
                        model, D = clf.fit(X, y)

                        if not isinstance(D, type(None)):
                            if D.ndim > 1:
                                D = D[:, self.c_choice]

                            L = np.where(D > 0, True, False)
                            R = np.where(D <= 0, True, False)

                            X_L_n = X[L].shape[0]
                            X_R_n = X[R].shape[0]

                            # Calculate Information Gain
                            if X_L_n > 0 and X_R_n > 0:
                                IG = purity_function(
                                    counts_sum, counts_prob, L, R, y, impurity, q
                                )

                                gains.append(IG)
                                hyperplane_list.append((model, L, R))
                                model_type.append(model.model_type)

                # Train a Neural Network
                if use_nnet:
                    if X.shape[0] >= nnet_min_samples:
                        for clf in [
                            ANNClassifier(
                                n_feat=max_features,
                                minority=minority_sz_nnet,
                            )
                        ]:
                            model, D = clf.fit(X, y)

                            if not isinstance(D, type(None)):
                                if D.ndim > 1:
                                    D = D[:, self.c_choice]

                                L = np.where(D > 0, True, False)
                                R = np.where(D <= 0, True, False)

                                X_L_n = X[L].shape[0]
                                X_R_n = X[R].shape[0]

                                # Calculate Information Gain
                                if X_L_n > 0 and X_R_n > 0:
                                    IG = purity_function(
                                        counts_sum, counts_prob, L, R, y, impurity, q
                                    )

                                    gains.append(IG)
                                    hyperplane_list.append((model, L, R))
                                    model_type.append(model.model_type)

                # Train Decision Tree Models
                if use_etc:
                    for clf in [
                        ETClassifier(
                            n_feat=max_features,
                            max_depth=etc_max_depth,
                            max_trees=etc_max_trees,
                        )
                    ]:
                        model, D = clf.fit(X, y)

                        if not isinstance(D, type(None)):
                            if D.ndim > 1:
                                D = D[:, self.c_choice]

                            L = np.where(D > 0, True, False)
                            R = np.where(D <= 0, True, False)

                            X_L_n = X[L].shape[0]
                            X_R_n = X[R].shape[0]

                            # Calculate Information Gain
                            if X_L_n > 0 and X_R_n > 0:
                                IG = purity_function(
                                    counts_sum, counts_prob, L, R, y, impurity, q
                                )

                                gains.append(IG)
                                hyperplane_list.append((model, L, R))
                                model_type.append(model.model_type)

                gains = np.asarray(gains)
                hyperplane_list = np.asarray(hyperplane_list, dtype="object")
                model_type = np.asarray(model_type, dtype=str)

                # Ensure that there is at least one set of splits that meets the minimum gain criteria
                gain_mask = np.where(gains >= min_gain, True, False)

                # Split the node
                if np.any(gain_mask):
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

                    # Append the output of the decision function to each dataframe
                    if use_cascade:
                        if isinstance(self.splitter, LMClassifier):
                            X_cascade = self.splitter.decision_function(X)

                            if X_cascade.ndim == 1:
                                X_cascade = X_cascade.reshape(-1, 1)

                        else:
                            X_cascade = self.splitter.predict_proba(X)

                        X_new = np.hstack((X, X_cascade))

                    else:
                        X_new = X

                    # Recursivly split
                    self.left = Node().get_split(
                        X_new[L],
                        y[L],
                        min_samples_in_leaf=min_samples_in_leaf,
                        max_depth=max_depth,
                        max_features=max_features,
                        min_gain=min_gain,
                        impurity=impurity,
                        q=q,
                        use_lm_l2=use_lm_l2,
                        use_lm_l1=use_lm_l1,
                        minority_sz_lm=minority_sz_lm,
                        use_nnet=use_nnet,
                        nnet_min_samples=nnet_min_samples,
                        minority_sz_nnet=minority_sz_nnet,
                        use_etc=use_etc,
                        etc_max_depth=etc_max_depth,
                        etc_max_trees=etc_max_trees,
                        N=X.shape[0],
                        current_depth=current_depth + 1,
                        use_oracle=use_oracle,
                        use_cascade=use_cascade,
                    )

                    self.right = Node().get_split(
                        X_new[R],
                        y[R],
                        min_samples_in_leaf=min_samples_in_leaf,
                        max_depth=max_depth,
                        max_features=max_features,
                        min_gain=min_gain,
                        impurity=impurity,
                        q=q,
                        use_lm_l2=use_lm_l2,
                        use_lm_l1=use_lm_l1,
                        minority_sz_lm=minority_sz_lm,
                        use_nnet=use_nnet,
                        nnet_min_samples=nnet_min_samples,
                        minority_sz_nnet=minority_sz_nnet,
                        use_etc=use_etc,
                        etc_max_depth=etc_max_depth,
                        etc_max_trees=etc_max_trees,
                        N=X.shape[0],
                        current_depth=current_depth + 1,
                        use_oracle=use_oracle,
                        use_cascade=use_cascade,
                    )

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
        q,
        use_oracle,
        use_lm_l2,
        use_lm_l1,
        minority_sz_lm,
        use_nnet,
        nnet_min_samples,
        minority_sz_nnet,
        use_etc,
        etc_max_depth,
        etc_max_trees,
        resampler,
        use_cascade,
    ):
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.q = q
        self.use_oracle = use_oracle
        self.use_lm_l2 = use_lm_l2
        self.use_lm_l1 = use_lm_l1
        self.minority_sz_lm = minority_sz_lm
        self.use_nnet = use_nnet
        self.nnet_min_samples = nnet_min_samples
        self.minority_sz_nnet = minority_sz_nnet
        self.use_etc = use_etc
        self.etc_max_depth = etc_max_depth
        self.etc_max_trees = etc_max_trees
        self.resampler = resampler
        self.use_cascade = use_cascade

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Increase diversity by resampling
        if isinstance(self.resampler, type(None)):
            X_re = X
            y_re = y

        else:
            self.resampler = clone(self.resampler)

            X_re, y_re = self.resampler.fit_resample(X, y)

        # Create the root node
        tree = Node()

        # Begin Splitting
        tree.get_split(
            X=X_re,
            y=y_re,
            min_samples_in_leaf=self.min_samples_in_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_gain=self.min_gain,
            impurity=self.impurity,
            q=self.q,
            use_lm_l2=self.use_lm_l2,
            use_lm_l1=self.use_lm_l1,
            minority_sz_lm=self.minority_sz_lm,
            use_nnet=self.use_nnet,
            nnet_min_samples=self.nnet_min_samples,
            minority_sz_nnet=self.minority_sz_nnet,
            use_etc=self.use_etc,
            etc_max_depth=self.etc_max_depth,
            etc_max_trees=self.etc_max_trees,
            N=X.shape[0],
            current_depth=1,
            use_oracle=self.use_oracle,
            use_cascade=self.use_cascade,
        )

        self.LMTree = tree

        self.all_nodes = self._get_all_nodes(self.LMTree)

        self.terminal_nodes = [x[0] for x in self.all_nodes if x[1] == 1]
        self.all_nodes = [x[0] for x in self.all_nodes]

        return self

    def _predict(self, X, current_node=None, samp_idx=None):
        final_predictions = []

        # Get a list of sample IDs if sample_index is not provided and set the node to the root of the tree
        if isinstance(samp_idx, type(None)):
            samp_idx = np.asarray([i for i in range(X.shape[0])])

            current_node = self.LMTree

        if current_node.terminal is False:
            # Determine where each sample goes
            D = current_node.splitter.decision_function(X)

            if D.ndim > 1:
                D = D[:, current_node.c_choice]

            L = np.where(D > 0, True, False)
            R = np.where(D <= 0, True, False)

            # Append decision function data
            if self.use_cascade:
                if isinstance(current_node.splitter, LMClassifier) or isinstance(
                    current_node.splitter, RandomOracle
                ):
                    C = current_node.splitter.decision_function(X)

                    if C.ndim == 1:
                        C = C.reshape(-1, 1)

                else:
                    C = current_node.splitter.predict_proba(X)

                X_new = np.hstack((X, C))

            else:
                X_new = X

            X_L = X_new[L]
            left = samp_idx[L]

            X_R = X_new[R]
            right = samp_idx[R]

            if left.shape[0] > 0:
                predictions_left = self._predict(X_L, current_node.left, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:
                predictions_right = self._predict(X_R, current_node.right, right)
                final_predictions.extend(predictions_right)

        elif current_node.terminal:
            predictions = current_node.label(X)
            predictions = np.asarray(
                [(samp_idx[i], prediction) for i, prediction in enumerate(predictions)]
            )

            return predictions

        return final_predictions

    def predict(self, X):
        if hasattr(self.resampler, "transform"):
            X_trf = self.resampler.transform(X)

        else:
            X_trf = X

        tree_predictions = self._predict(X_trf)

        tree_predictions = [(int(entry[0]), entry[1]) for entry in tree_predictions]
        tree_predictions.sort()

        return np.asarray(tree_predictions)[:, 1]

    def score(self, X, y):
        score = balanced_accuracy_score(y, self.predict(X))

        return score

    def _get_all_nodes(self, node):
        node_list = set()

        if node.terminal is False:
            node_list.update([(node.node_id, 0)])

            node_list = node_list.union(self._get_all_nodes(node.left))
            node_list = node_list.union(self._get_all_nodes(node.right))

        elif node.terminal:
            node_list.update([(node.node_id, 1)])

            return node_list

        return node_list

    def _proximity(self, X, current_node=None, samp_idx=None):
        final_predictions = []

        # Get a list of sample IDs if sample_index is not provided and set the node to the root of the tree
        if isinstance(samp_idx, type(None)):
            samp_idx = np.asarray([i for i in range(X.shape[0])])

            current_node = self.LMTree

        # Check if the node is a terminal node
        if current_node.terminal is False:
            # Determine where each sample goes
            D = current_node.splitter.decision_function(X)

            if D.ndim > 1:
                D = D[:, current_node.c_choice]

            L = np.where(D > 0, True, False)
            R = np.where(D <= 0, True, False)

            # Append decision function data
            if self.use_cascade:
                if isinstance(current_node.splitter, LMClassifier) or isinstance(
                    current_node.splitter, RandomOracle
                ):
                    C = current_node.splitter.decision_function(X)

                    if C.ndim == 1:
                        C = C.reshape(-1, 1)

                else:
                    C = current_node.splitter.predict_proba(X)

                X_new = np.hstack((X, C))

            else:
                X_new = X

            X_L = X_new[L]
            left = samp_idx[L]

            X_R = X_new[R]
            right = samp_idx[R]

            if left.shape[0] > 0:
                predictions_left = self._proximity(X_L, current_node.left, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:
                predictions_right = self._proximity(X_R, current_node.right, right)
                final_predictions.extend(predictions_right)

        elif current_node.terminal:
            return [(entry, current_node.node_id) for entry in samp_idx]

        return final_predictions

    def _proximity_path(self, X, current_node=None, samp_idx=None):
        final_predictions = []

        # Get a list of sample IDs if sample_index is not provided and set the node to the root of the tree
        if isinstance(samp_idx, type(None)):
            samp_idx = np.asarray([i for i in range(X.shape[0])])

            current_node = self.LMTree

        # Check if the node is a terminal node
        if current_node.terminal is False:
            # Determine where each sample goes
            D = current_node.splitter.decision_function(X)

            if D.ndim > 1:
                D = D[:, current_node.c_choice]

            L = np.where(D > 0, True, False)
            R = np.where(D <= 0, True, False)

            # Append decision function data
            if self.use_cascade:
                if isinstance(current_node.splitter, LMClassifier) or isinstance(
                    current_node.splitter, RandomOracle
                ):
                    C = current_node.splitter.decision_function(X)

                    if C.ndim == 1:
                        C = C.reshape(-1, 1)

                else:
                    C = current_node.splitter.predict_proba(X)

                X_new = np.hstack((X, C))

            else:
                X_new = X

            X_L = X_new[L]
            left = samp_idx[L]

            X_R = X_new[R]
            right = samp_idx[R]

            if left.shape[0] > 0:
                final_predictions.extend(
                    [(entry, current_node.node_id) for entry in samp_idx[L]]
                )
                predictions_left = self._proximity_path(X_L, current_node.left, left)
                final_predictions.extend(predictions_left)

            if right.shape[0] > 0:
                final_predictions.extend(
                    [(entry, current_node.node_id) for entry in samp_idx[R]]
                )
                predictions_right = self._proximity_path(X_R, current_node.right, right)
                final_predictions.extend(predictions_right)

        elif current_node.terminal:
            return [(entry, current_node.node_id) for entry in samp_idx]

        return final_predictions

    def proximity(self, X, prox_type="path"):
        if hasattr(self.resampler, "transform"):
            X_trf = self.resampler.transform(X)

        else:
            X_trf = X

        if prox_type == "terminal":
            tree_predictions = self._proximity(X_trf)

            tree_predictions.sort()

            col_dict = {col: i for i, col in enumerate(self.terminal_nodes)}

            emb_matrix = np.zeros(
                shape=(X.shape[0], len(self.terminal_nodes)), dtype=np.ushort
            )

            for entry in tree_predictions:
                row = entry[0]
                col = col_dict[entry[1]]

                emb_matrix[row, col] = 1

            return emb_matrix

        elif prox_type == "path":
            tree_predictions = self._proximity_path(X_trf)

            emb_matrix = {}
            for sample in tree_predictions:
                if sample[0] not in emb_matrix:
                    emb_matrix[sample[0]] = set()

                emb_matrix[sample[0]].add(sample[1])

            return emb_matrix
