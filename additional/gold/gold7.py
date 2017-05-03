import numpy as np

from matplotlib.mlab import find
import matplotlib.pyplot as pl

from cvxopt import sparse, matrix, spmatrix, solvers
from sklearn.datasets import make_blobs, make_classification
from scipy.spatial.distance import cdist


def eye(n):
    return spmatrix(1, range(n), range(n))


def visualize(clf, X, y):
    border = .5
    h = .02

    x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
    y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    pl.figure(1, figsize=(8, 6))
    pl.pcolormesh(xx, yy, z_class, cmap=pl.cm.summer, alpha=0.3)

    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    pl.contour(xx, yy, z_dist, [0.0], colors='black')
    pl.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    y_pred = clf.predict(X)

    ind_support = clf.support_
    ind_correct = list(set(find(y == y_pred)) - set(ind_support))
    ind_incorrect = list(set(find(y != y_pred)) - set(ind_support))
    pl.scatter(X[ind_correct, 0], X[ind_correct, 1], c=y[ind_correct],
               cmap=pl.cm.summer, alpha=0.9)
    pl.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=y[ind_incorrect],
               cmap=pl.cm.summer, alpha=0.9, marker='*', s=50)
    pl.scatter(X[ind_support, 0], X[ind_support, 1], c=y[ind_support],
               cmap=pl.cm.summer, alpha=0.9, linewidths=1.8, s=40)

    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())


class LinearSVM:
    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        (n, k) = X.shape
        n_vars = k + 1 + n
        iw = list(range(k))
        iw0 = k
        iksi = list(range(iw0 + 1, n_vars))
        P = spmatrix(1, iw, iw, (n_vars, n_vars))
        q = matrix([0] * (k + 1) + [self.C] * n, (n_vars, 1), tc='d')

        G_0 = spmatrix(-1, range(n), iksi, (n, n_vars))
        G_1 = sparse([
            [matrix(-y[:, np.newaxis] * X)],
            [matrix(-y)],
            [-eye(n)]
        ])
        G = sparse([G_0, G_1])
        h = matrix([0] * n + [-1] * n, (2 * n, 1), tc='d')

        result = solvers.qp(P, q, G, h)
        alpha = np.array(result['z'][n:]).reshape((n,))
        self.support_ = alpha > 1e-4
        print(self.support_)
        self.w = np.array(result['x'][:k]).reshape((k,))
        self.w0 = result['x'][k]

    def decision_function(self, X):
        return X.dot(self.w) + self.w0

    def predict(self, X):
        return np.sign(self.decision_function(X))


class KernelSVM:
    def __init__(self, C, kernel=None, sigma=1.0, degree=2):
        self.C = C
        self.kernel = lambda X1, X2: kernel(X1, X2, sigma=sigma, degree=degree)
        self.sigma = sigma
        self.degree = degree

    def fit(self, X, y):
        n, k = X.shape
        XX = self.kernel(X, X)
        P = matrix(np.outer(y, y) * XX)
        q = matrix(-1, (n, 1), tc='d')
        G = sparse([eye(n), -eye(n)])
        h = matrix([self.C] * n + [0] * n, tc='d')
        A = matrix(y, (1, n), tc='d')
        b = matrix(0, (1, 1), tc='d')
        result = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(result['x']).reshape((n,))
        self.support_ = supp = alpha > 1e-4
        self.SX, self.sy, self.salpha = X[supp], y[supp], alpha[supp]
        SXX = XX[supp, :][:, supp]
        self.w0 = (self.sy - (self.salpha * self.sy * SXX).sum(axis=1)).mean()

    def decision_function(self, X):
        return (self.salpha * self.sy * self.kernel(X, self.SX)).sum(axis=1) + self.w0

    def predict(self, X):
        return np.sign(self.decision_function(X))


def linear_kenel(X1, X2, **q):
    return X1.dot(X2.T)


def poly_kernel(X1, X2, degree=3, **q):
    return (X1.dot(X2.T) + 1) ** degree


def gaussian_kernel(X1, X2, sigma=2, **q):
    return np.exp(-sigma * cdist(X1, X2, 'euclid') ** 2)


# X, y = make_blobs(n_samples=1000, centers=2)
X, y = make_classification(n_samples=500,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0)
y = 2 * y - 1

for cls, title in [
    (LinearSVM(C=1), 'linear'),
    (KernelSVM(C=1, kernel=linear_kenel), 'linear kernel'),
    (KernelSVM(C=1, kernel=poly_kernel, degree=2), 'quadratic kernel'),
    (KernelSVM(C=1, kernel=gaussian_kernel), 'gaussian kernel')]:
    cls.fit(X, y)
    visualize(cls, X, y)
    pl.title(title)
    pl.show()
