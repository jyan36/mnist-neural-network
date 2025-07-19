import unittest

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from load_mnist import load_data

import numpy as np


class TestLoadMNIST(unittest.TestCase):
    def test_data_shapes_and_types(self):
        X_train, y_train, X_test, y_test = load_data()

        self.assertEqual(X_train.shape, (60000, 784))
        self.assertEqual(y_train.shape, (60000, 10))
        self.assertEqual(X_test.shape, (10000, 784))
        self.assertEqual(y_test.shape, (10000, 10))

        self.assertTrue(np.issubdtype(X_train.dtype, np.floating))
        self.assertTrue(np.issubdtype(y_train.dtype, np.floating))

        self.assertTrue(np.max(X_train) <= 1.0)
        self.assertTrue(np.min(X_train) >= 0.0)


if __name__ == "__main__":
    unittest.main()
