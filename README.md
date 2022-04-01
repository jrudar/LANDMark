### LANDMark
Implementation of a decision tree ensemble which splits each node using learned linear and non-linear functions.

### Install
Once downloaded, go to the LANDMark directory and type:
    pip install .
    
### Class Parameters
    n_estimators: int, default = 64
        The number of trees used to construct the ensemble.

    min_samples_in_leaf: int, default = 5
        The minimum number of samples in each leaf to proceed to cutting.
        
    max_depth: int, default = -1
        The maximum depth of the tree. '-1' implies that trees will fully
        grow until a stopping criterion is met.
        
    max_features: float, default = 0.80
        The maximum features used to train each model at each node. These
        features are randomly selected at each node.
        
    min_gain: float, default = 0
        The minimum gain needed to split a node.
        
    impurity: str, default = "gain"
        The method by which LANDMark calculates the purity of each split.
        Currently, only information gain is used.
        
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
        
    use_etc: bool, default = True
        Specifies if Extremely Randomized Tree models should be trained. Could
        be used as an alternative to neural network models.
        
    bootstrap: bool, default = False
        Specifies if each tree is built using a bootstrap sample.
        
    n_jobs: int, default = 4
        The number of processes used to create the LANDMark model.
            
### Fit Parameters
        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (features, taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.
        
### Example Usage
        from LANDMark import LANDMarkClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification()
        
        clf = LANDMarkClassifier().fit(X, y)

### References

    Rudar, J., Porter, T.M., Wright, M. et al. LANDMark: an ensemble approach to the 
    supervised selection of biomarkers in high-throughput sequencing data. 
    BMC Bioinformatics 23, 110 (2022). https://doi.org/10.1186/s12859-022-04631-z

    Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: 
    Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–30. 

    Kuncheva LI, Rodriguez JJ. Classifier ensembles with a random linear oracle. 
    IEEE Transactions on Knowledge and Data Engineering. 2007;19(4):500–8. 
    
    Geurts P, Ernst D, Wehenkel L. Extremely Randomized Trees. Machine Learning. 2006;63(1):3–42. 

