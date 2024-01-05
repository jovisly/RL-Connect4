from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def get_data(seed=777):
    d = load_breast_cancer()
    X = d.data
    y = d.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test

