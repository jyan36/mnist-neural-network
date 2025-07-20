import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, X):
        self.z = []
        self.a = []

        a = X
        self.a.append(a)

        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self.relu(z)
            self.a.append(a)

        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        a = self.softmax(z)
        self.a.append(a)

        return a

    def backward(self, X, y, output, lr):
        m = y.shape[0]
        grads_w = [0] * self.num_layers
        grads_b = [0] * self.num_layers

        delta = (output - y) / m

        for i in reversed(range(self.num_layers)):
            grads_w[i] = np.dot(self.a[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_deriv(self.z[i - 1])

        for i in range(self.num_layers):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X, y, epochs, lr, batch_size=64):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                output = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_batch, output)
                epoch_loss += loss * X_batch.shape[0]
                self.backward(X_batch, y_batch, output, lr)

            epoch_loss /= n_samples
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
