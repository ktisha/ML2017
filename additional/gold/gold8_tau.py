import numpy as np

MAX_ITER = 10**5

class NormalLR:
    def __init__(self, *, tau=None):
        self.tau = tau

    def fit(self, X, y):
        if self.tau:
            F_plus = np.linalg.inv(X.T.dot(X) + self.tau * np.eye(len(X.T))).dot(X.T)   # <-- HERE
        else:
            F_plus = np.linalg.pinv(X)

        self.weights = F_plus.dot(y)
        return self

    def predict(self, X):
        return X.dot(self.weights)


class GradientLR(NormalLR):
    def __init__(self, *, alpha, tau=0):
        self.alpha = alpha
        self.tau = tau
        self.threshold = alpha / 100

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.random.random(n_features) - 0.5

        for i in range(MAX_ITER):
            grad = np.dot((X.dot(self.weights) - y), X) / n_samples

            grad += self.tau * self.weights / n_samples      # <-- HERE

            difference = self.alpha * grad

            if np.linalg.norm(difference) < self.threshold:
                break

            self.weights -= difference

        return self

