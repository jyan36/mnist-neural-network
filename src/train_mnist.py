from load_mnist import load_data
from neural_network import NeuralNetwork
import numpy as np

X_train, y_train, X_test, y_test = load_data()
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
nn = NeuralNetwork(input_size=784, hidden_sizes=[128, 64, 32], output_size=10)
nn.train(X_train, y_train, epochs=100, lr=0.05)

preds = nn.predict(X_test)
labels = np.argmax(y_test, axis=1)
accuracy = np.mean(preds == labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
