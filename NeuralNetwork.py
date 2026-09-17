import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]

            W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            b = np.zeros((1, output_size))

            self.weights.append(W)
            self.biases.append(b)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        activation = X
        self.z = []
        self.activations = [X]

        for i in range(len(self.weights)):
            z = (np.dot(activation, self.weights[i]) + self.biases[i])
            self.z.append(z)

            if i < len(self.weights) - 1:
                activation = self.ReLU(z)
            else:
                activation = self.softmax(z)

            self.activations.append(activation)
        return activation

    def backward(self, X, y):
        m = X.shape[0]
        NumberOfLayers = len(self.weights)

        dW = [None] * NumberOfLayers
        db = [None] * NumberOfLayers
        dz = self.activations[-1] - y

        for i in range(NumberOfLayers - 1, -1, -1):
            pre_activation = self.activations[i]
            dW[i] = np.dot(pre_activation.T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self.ReLU_derivative(self.z[i-1])

        for i in range(NumberOfLayers):
            self.weights[i] -= self.lr * dW[i]
            self.biases[i] -= self.lr * db[i]

    def fit(self, X, y, batch_size=128):
        m = X.shape[0]
        for epoch in range(self.epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            y_predict = self.forward(X)
            predict = np.argmax(y_predict, axis=1)
            target = np.argmax(y, axis=1)
            accuracy = np.mean(predict == target)
            print(f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {accuracy:.4f}")

    def predict(self, X):
        y_predict = self.forward(X)
        return np.argmax(y_predict, axis=1)

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test_raw = X_test.copy()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    y_train_encoded = np.eye(10)[y_train]
    y_test_encoded = np.eye(10)[y_test]

    model = NeuralNetwork(layer_sizes=[784, 128, 32, 10], learning_rate=0.1, epochs=10)
    model.fit(X_train, y_train_encoded, batch_size=128)

    test_predicts = model.predict(X_test)
    test_targets = np.argmax(y_test_encoded, axis=1)
    accuracy = np.mean(test_predicts == test_targets)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%\n")

    NumberOfImages = 100
    indices = np.random.choice(len(X_test), size=NumberOfImages, replace=False)

    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(indices):
        plt.subplot(10, 10, i + 1)
        plt.imshow(X_test_raw[idx], cmap='gray')
        predict = test_predicts[idx]
        true_label = test_targets[idx]
        color = 'green' if predict == true_label else 'red'
        plt.title(f"(Pred: {predict} | True: {true_label})", color=color, fontsize=7)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()