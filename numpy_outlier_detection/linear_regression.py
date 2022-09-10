import numpy as np


class LinearRegressionNumpy:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.coef = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coef[0]
        self.coef = self.coef[1:]
        return self

    def predict(self, X):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.coef) + self.intercept
