.. -*- mode: rst -*-

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.18
.. |JobLibMinVersion| replace:: 0.17.0
.. |PandasMinVersion| replace:: 1.0.3
.. |TensorflowMinVersion| replace:: 2.6.0
.. |TensorflowAddonsMinVersion| replace:: 0.14.0
.. |NumbaMinVersion| replace:: 0.44.1
.. |shapMinVersion| replace:: 0.34.0


**LANDMark** is a Python package which implements a multivariate decision tree ensemble.

Dependencies
------------

LANDMark requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Tensorflow (>= |TensorflowMinVersion|)
- Tensorflow Addons (>= |TensorflowAddonsMinVersion|)
- Numba (>= |NumbaMinVersion|)
- shap (>= |shapMinVersion|)

Installation
------------
To install LANDMark, please download the installation files and install by typing:

``python setup.py install``

Training and Making Predictions Using a LANDMark Model
------------
To train and use a LANDMark model, simply import the module and call the fit, predict, and predict_proba commands:

```

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from LANDMark import LANDMarkClassifier

X, y = load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

model = LANDMarkClassifier().fit(X_train, y_train)

class_labels = model.predict(X_test)

class_probabilities = model.predict_proba(X_test)

```
