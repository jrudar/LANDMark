from LANDMark import LANDMarkClassifier
from LANDMark.lm_nnet_clfs import ANNClassifier
from LANDMark.lm_linear_clfs import LMClassifier
from LANDMark.lm_oracle_clfs import RandomOracle
from LANDMark.lm_dtree_clfs import ETClassifier

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from pathlib import Path

dirpath = Path(__file__).parent

# Tests the transformer modules
def test_landmark():

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
    clf = LANDMarkClassifier(n_estimators = 16, n_jobs = 2, min_samples_in_leaf = 2)
    clf.fit(X_train, y_train)

    # Make a prediction
    predictions = clf.predict(X_test)

    # Score
    BAcc = clf.score(X_test, y_test)
    assert BAcc >= 0.85

    # Get proximity
    prox = clf.proximity(X_train)

def test_models():

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

    clf, _ = ANNClassifier().fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7

    clf, _ = ETClassifier().fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf = RandomOracle().fit(X_train, y_train)
    p = clf.predict(X_test)
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="lr_l2").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="sgd_l2").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="ridge").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="lsvc").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="lr_l1").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)

    clf, _ = LMClassifier(model_type="sgd_l1").fit(X_train, y_train)
    p = clf.predict(X_test)
    BAcc = balanced_accuracy_score(y_test, p)
    assert BAcc > 0.7
    D = clf.decision_function(X_test)


test_models()

test_landmark()

