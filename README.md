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

Examples of how to use `LANDMark` can be found [here](notebooks/README.md).

## Contributing

To contribute to the development of `LANDMark` please read our [contributing guide](docs/CONTRIBUTING.md)


### References

    Rudar, J., Porter, T.M., Wright, M., Golding G.B., Hajibabaei, M. LANDMark: an ensemble 
    approach to the supervised selection of biomarkers in high-throughput sequencing data. 
    BMC Bioinformatics 23, 110 (2022). https://doi.org/10.1186/s12859-022-04631-z

    Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: 
    Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–30. 

    Kuncheva LI, Rodriguez JJ. Classifier ensembles with a random linear oracle. 
    IEEE Transactions on Knowledge and Data Engineering. 2007;19(4):500–8. 
    
    Geurts P, Ernst D, Wehenkel L. Extremely Randomized Trees. Machine Learning. 2006;63(1):3–42. 

