import numpy as np


def sigmoid_loss(M):
    """
    :type M: numpy.core.multiarray.ndarray
    """
    return 2 / (1 + np.exp(M)), -2 * np.exp(M) / (np.exp(M) + 1) ** 2


class GradientDescent:
    def __init__(self, *, alpha, threshold=1e-2, loss=sigmoid_loss):
        self.weights = []
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss

    def fit(self, X, y):
        n = X.shape[1]
        self.weights = np.random.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        errors = []

        while True:
            M = X.dot(self.weights) * y
            loss, derivative = self.loss(M)

            grad_q = np.sum((derivative.T * (X.T * y)).T, axis=0)

            tmp = self.weights - self.alpha * grad_q

            errors.append(np.sum(loss))
            if np.linalg.norm(self.weights - tmp) < self.threshold:
                break
            self.weights = tmp
        return errors

    def predict(self, X):
        return np.sign(X.dot(self.weights))


class SGD:
    def __init__(self, *, alpha, loss=sigmoid_loss, k=1, n_iter=100):
        self.k = k
        self.n_iter = n_iter
        self.alpha = alpha
        self.loss = loss

    def fit(self, X, y):
        n = X.shape[1]
        self.weights = np.random.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        errors = []

        q, _ = self.calc_grad(self.weights, X, y)
        eta = 1 / len(y)
        for i in range(self.n_iter):
            batch_index = np.random.choice(len(y), self.k)
            loss, grad = self.calc_grad(self.weights, X[batch_index], y[batch_index])
            q = q * (1 - eta) + loss * eta
            errors.append(q)

            self.weights -= self.alpha * grad

        return errors

    def calc_grad(self, w, X, y):
        M = X.dot(w) * y
        l, l_der = self.loss(M)
        grad = X.T * l_der * y
        return l.mean(), grad.mean(axis=1)

    def predict(self, X):
        return np.sign(X.dot(self.weights))
