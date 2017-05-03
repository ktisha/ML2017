import numpy as np
from numpy import random
import cv2


def ceuclidean(A, B):
    assert A.ndim == B.ndim == 2
    D = np.empty((len(A), len(B)), dtype=np.float64)
    for i, Ai in enumerate(A):
        D[i, :] = np.sqrt(np.square(Ai - B).sum(axis=1))
    return D


def init_centers(X, n_clusters):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))
    centers[0] = X[random.choice(n_samples)]
    for i in range(1, n_clusters):
        [D] = np.square(ceuclidean(centers[i - 1:i], X))
        D /= D.sum()
        centers[i] = X[random.choice(n_samples, p=D)]
    return centers


def kmeans(X, n_clusters):
    centers = init_centers(X, n_clusters)
    y = None
    while True:
        # distances = ceuclidean(centers, X)
        distance = lambda Ai, X: np.sqrt(np.square(Ai - X).sum(axis=1))

        distances = np.empty((len(centers), len(X)), dtype=np.float64)

        for i, Ai in enumerate(centers):
            distances[i, :] = distance(Ai, X)

        new_y = distances.argmin(axis=0)
        if np.array_equal(y, new_y):
            break
        y = new_y
        for i in range(n_clusters):
            centers[i] = X[y == i].mean(axis=0)
    return centers, y


def plot_colors(hist, centroids):
    height, width = (50, 200)
    bar = np.zeros((height, width, 3), dtype='uint8')
    start_x = 0

    for percent, color in zip(hist, centroids):
        end_x = start_x + width * percent
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), height), color.astype('uint8').tolist(), -1)
        start_x = end_x

    return bar


def recolor(image, n_colors):
    c, mu = kmeans(image.astype(np.int64), n_colors)
    cv2.imwrite('superman-batman-16-colors.png', mu[c].reshape(1024, 768, 3))


def read_image(path='superman-batman.png'):
    return cv2.imread(path).reshape(-1, 3)
