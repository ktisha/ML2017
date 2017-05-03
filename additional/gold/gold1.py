import numpy as np
from scipy.stats import mode

# X_train = (l objects, n features) 146, 13
# X_test = (m objects, n features)   42, 13


def train_test_split(X, y, ratio):
    mask = np.random.uniform(size=len(y)) < ratio
    return X[mask], y[mask], X[~mask], y[~mask]


def knn(X_train, y_train, X_test, k, dist):
    distances = dist(X_train, X_test)   # (l, m)
    print(distances.shape)
    ids = distances.argsort(axis=1)[:k, :]  # (k, m)
    k_min_values = y_train[ids]
    result = mode(k_min_values, axis=0)
    return result[0][0]


def euclidean(A, B):
    D = np.empty((len(A), len(B)))
    for i, Ai in enumerate(A):
        D[i, :] = np.sqrt(np.square(Ai - B).sum(axis=1))
    return D


def print_precision_recall(y_pred, y_test):
    for c in np.unique(y_test):
        tp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] == c])
        fp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] != c])
        fn = len([i for i in range(len(y_pred)) if y_pred[i] != y_test[i] and y_test[i] != c])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")


def loo(X_train, y_train, dist, k):
    l = []
    for i in range(len(y_train)):
        if knn(np.array(X_train[:i] + X_train[i + 1:]),
               np.array(y_train[:i] + y_train[i + 1:]),
               np.array(X_train[i]),
               k, dist) != y_train[i]:
            l.append(i)
    return len(l)


def loocv(X_train, y_train, dist):
    loo_list = []
    for k in range(1, len(y_train)):
        loo_list.append(loo(X_train, y_train, dist, k))
    opt_k = loo_list.index(min(loo_list)) + 1
    return opt_k

if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')

    X, y = wines[:, 1:], wines[:, 0]
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)

    result = knn(X_train, y_train, X_test, 5, euclidean)

