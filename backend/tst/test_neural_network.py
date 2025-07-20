import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(input_size=4, hidden_sizes=[5], output_size=3)
        self.X = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.y = np.array([[0, 1, 0]], dtype=np.float64)  # One-hot, float type

    def test_forward_output_shape(self):
        output = self.nn.forward(self.X)
        self.assertEqual(output.shape, (1, 3))

    def test_forward_values_range(self):
        output = self.nn.forward(self.X)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))

    def test_backward_shapes(self):
        output = self.nn.forward(self.X)
        weights_before = [w.copy() for w in self.nn.weights]
        self.nn.backward(self.X, self.y, output, lr=0.01)
        changed = any(
            not np.array_equal(wb, wa)
            for wb, wa in zip(weights_before, self.nn.weights)
        )
        self.assertTrue(changed, "Weights should be updated during backward pass")

    def test_predict(self):
        pred = self.nn.predict(self.X)
        self.assertEqual(pred.shape, (1,))
        self.assertTrue(0 <= pred[0] < 3)

    def test_training_reduces_loss(self):
        X = np.random.rand(10, 4)
        y = np.zeros((10, 3))
        y[np.arange(10), np.random.randint(0, 3, 10)] = 1

        loss_before = np.mean(np.square(y - self.nn.forward(X)))
        self.nn.train(X, y, epochs=50, lr=0.1, batch_size=10)
        loss_after = np.mean(np.square(y - self.nn.forward(X)))
        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
