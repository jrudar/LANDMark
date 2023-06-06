## Overview of Section

This section provides an overview of the `LANDMark` and the different parameters
of the `LANDMark` class and its methods.

## Class

    class LANDMark.LANDMark(n_estimators, min_samples_in_leaf, max_depth, max_features, min_gain, impurity, q, use_oracle, 
        use_lm_l2, use_lm_l1, use_nnet, nnet_min_samples, use_etc, etc_max_depth = 5, etc_max_trees = 128, resampler = None,
        use_cascade = False, n_jobs = 4)

### Parameters

### Class Parameters
    n_estimators: int, default = 64
        The number of trees used to construct the ensemble.

    min_samples_in_leaf: int, default = 5
        The minimum number of samples in each leaf to proceed to cutting.
        
    max_depth: Optional[int], default = None
        The maximum depth of the tree. 'None' implies that trees will fully
        grow until a stopping criterion is met.
        
    max_features: float, default = 0.80
        The maximum features used to train each model at each node. These
        features are randomly selected at each node.
        
    min_gain: float, default = 0.0
        The minimum gain needed to split a node.
        
    impurity: str, default = "gain"
        The method by which LANDMark calculates the purity of each split.
        Currently the options are 'gain', 'tsallis', 'gain-ratio', and 
        'tsallis-gain-ratio'.
        
    q: float, default = 1.5
        Parameter of the Tsallis entropy function. Smaller values may be more
        important when there are many classes. A value of 1.0 corresponds to
        information gain using Shannon entropy while a value of 2.0 corresponds
        to the gini index. Only used when impurity is set to 'tsallis' or
        'tsallis-gain-ratio'.
        
    use_oracle: bool, default = True
        Specifies if a random linear oracle should be used as the initial
        split.
        
    use_lm_l2: bool, default = True
        Specifies if linear models using L2 regularization should be trained.
        
    use_lm_l1: bool, default = True
        Specifies if linear models using L1 regularization should be trained.
        
    use_nnet: bool, default = True
        Specifies if neural network models should be trained.
        
    nnet_min_samples: int, default = 32
        The minimum number of samples needed to train a neural network model.
        Only used if 'use_nnet' is True.
        
    use_etc: bool, default = True
        Specifies if Extremely Randomized Tree models should be trained. Could
        be used as an alternative to neural network models.

    etc_max_depth: int, default = 5
        Specifies the maximum depth of each ExtraTreesClassifier. Only used
        if 'use_etc' is set to True.

    etc_max_trees: int, default = 128
        Specifies the maximum depth of trees used to train each ExtraTreesClassifier. 
        Only used if 'use_etc' is set to True.
                
    resampler: The resampling object. Cloning of the object must be possible and,
        at a minimum, the object must have a 'fit_resample(X, y)' method. The
        resampling object can also have a 'transform(X)' method if a user-defined
        transformation occurs during fitting.

    use_cascade: bool, default = False
        This parameter extends 'X' using the information returned by the best decision
        function within each node. By doing this, information about the split is
        retained and has the potential to be used within deeper nodes of the tree.
        The inspiration for this idea comes from: https://www.ijcai.org/proceedings/2017/0497.pdf
        
    n_jobs: int, default = 4
        The number of processes used to create the LANDMark model.
            

### Attributes

    estimators_: Ensemble Object
        The collection of individual LANDMark trees.

    classes_: np.ndarray
        The classes present in the training data.


### Methods

    fit(X, y)
        Fits a `LANDMark` model.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted LANDMark object.

    predict_proba(X)
        Predicts class probabilities.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        A np.ndarray of shape (m, p) where 'm' is the number of samples in X and
        'p' is the number of classes.

    predict(X)
        Predicts class labels.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        A np.ndarray of shape (m,) where 'm' is the number of samples

    score(X, y)
        Calculates the balanced accuracy score.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        score: A float which is the balanced accuracy score.

    proximity(X)
        Transforms X into the high-dimensional LANDMark embedding.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (leaves of the LANDMark embedding)
