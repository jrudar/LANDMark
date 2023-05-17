from LANDMark import LANDMarkClassifier

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
    clf = LANDMarkClassifier(n_estimators = 8, n_jobs = 8)
    clf.fit(X_train, y_train)

    # Make a prediction
    predictions = clf.predict(X_test)

    # Score
    BAcc = clf.score(X_test, y_test)
    print(BAcc)

    # Get proximity
    prox = clf.proximity(X_train)
    prox = clf.proximity(X_test)

test_landmark()

