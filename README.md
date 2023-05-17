### LANDMark

[![CI](https://github.com/jrudar/LANDMark/actions/workflows/ci.yml/badge.svg)](https://github.com/jrudar/LANDMark/actions/workflows/ci.yml)

Implementation of a decision tree ensemble which splits each node using learned linear and non-linear functions.

### Install
From PyPI:

```bash
pip install LANDMarkClassifier
```

From source:

```bash
git clone https://github.com/jrudar/LANDMark.git
cd LANDMark
pip install .
# or create a virtual environment
python -m venv venv
source venv/bin/activate
pip install .
```

## Interface

An overview of the API can be found [here](docs/API.md).

## Usage and Examples

Comming Soon

## Contributing

To contribute to the development of `LANDMark` please read our [contributing guide](docs/CONTRIBUTING.md)

## Basic Usage

    from LANDMark import LANDMarkClassifier

    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Create the dataset
    X, y = load_wine(return_X_y = True)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Standardize
    X_trf = StandardScaler()
    X_trf.fit(X_train)

    X_train = X_trf.transform(X_train)
    X_test = X_trf.transform(X_test)

    # Setup a LANDMark model and fit
    clf = LANDMarkClassifier()
    clf.fit(X_train, y_train)

    # Make a prediction
    predictions = clf.predict(X_test)

### Specal Notes

Starting with TensorFlow 2.11, GPU support on Windows 10 and higher requires Windows WSL2.
See: https://www.tensorflow.org/install/pip

### References

    Rudar, J., Porter, T.M., Wright, M., Golding G.B., Hajibabaei, M. LANDMark: an ensemble 
    approach to the supervised selection of biomarkers in high-throughput sequencing data. 
    BMC Bioinformatics 23, 110 (2022). https://doi.org/10.1186/s12859-022-04631-z

    Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: 
    Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–30. 

    Kuncheva LI, Rodriguez JJ. Classifier ensembles with a random linear oracle. 
    IEEE Transactions on Knowledge and Data Engineering. 2007;19(4):500–8. 
    
    Geurts P, Ernst D, Wehenkel L. Extremely Randomized Trees. Machine Learning. 2006;63(1):3–42. 

