import numpy as np
from tensorflow.keras.datasets import mnist


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, 28 * 28)
    X_test = (X_test / 255.0).reshape(-1, 28 * 28)

    def one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return X_train, y_train, X_test, y_test
