import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def train(self, X, y, n_iter=100, learning_rate=1):
        X = X.reshape(X.shape + (1,))
        data = np.array(list(zip(X, y)))

        for j in range(n_iter):
            np.random.shuffle(data)

            gradient_b = [np.zeros(b.shape) for b in self.biases]
            gradient_w = [np.zeros(w.shape) for w in self.weights]

            for x, y in data:
                delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
                gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
                gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

            #: :type : list[numpy.core.multiarray.ndarray]
            self.weights = [w - (learning_rate / len(data)) * nw for w, nw in zip(self.weights, gradient_w)]

            #: :type : list[numpy.core.multiarray.ndarray]
            self.biases = [b - (learning_rate / len(data)) * nb for b, nb in zip(self.biases, gradient_b)]

    def feedforward(self, X):
        for b, w in zip(self.biases, self.weights):
            X = sigmoid(np.dot(w, X) + b)
        return X

    def backpropagation(self, X, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = X
        activations = [X]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        for layer_index in range(2, self.num_layers):
            z = zs[-layer_index]
            spv = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer_index + 1].T, delta) * spv

            gradient_b[-layer_index] = delta
            gradient_w[-layer_index] = np.dot(delta, activations[-layer_index - 1].T)

        return gradient_b, gradient_w

    def predict(self, X):
        X = X.reshape(X.shape + (1,))
        result = [np.argmax(self.feedforward(x)) for x in X]
        return result


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
