import numpy as np
from sklearn.cross_validation import train_test_split

def mse(actual, pred):
    """
    :type actual: numpy.core.multiarray.ndarray
    """

    return ((actual - pred) ** 2).mean()

MAX_ITER = 10**5

class NormalLR:
    def fit(self, X, y):
        F_plus = np.linalg.pinv(X)
        self.weights = F_plus.dot(y)
        return self

    def predict(self, X):
        return np.dot(X, self.weights)


class GradientLR(NormalLR):
    def __init__(self, alpha):
        self.alpha = alpha
        self.threshold = alpha / 100

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.empty(n_features)

        for i in range(MAX_ITER):
            grad = np.dot((X.dot(self.weights) - y), X) / n_samples
            difference = self.alpha * grad

            if np.linalg.norm(difference) < self.threshold:
                break

            self.weights -= difference

        return self


data = np.genfromtxt('boston.csv', delimiter=',', skip_header=15)
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)
