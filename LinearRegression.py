from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class LinearRegression:
    def __init__(self, max_iters=1000, learningRate=0.001):
        self.max_iters = max_iters
        self.lr = learningRate
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        NumberOfSamples, NumberOfFeatures = X.shape
        self.weight = np.zeros(NumberOfFeatures)
        self.bias = 0

        for _ in range(self.max_iters):
            y_predict = np.dot(X, self.weight) + self.bias
            dw = (1 / NumberOfSamples) * np.dot(X.T, y_predict - y)
            db = (1 / NumberOfSamples) * np.sum(y_predict - y)
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predict = np.dot(X, self.weight) + self.bias
        return y_predict

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression(max_iters=1000, learningRate=0.01)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    mse = np.mean((y_predict - y_test) ** 2)

    print(f"MSE: {mse:.4f}")
    for i in range(10):
        print(
            f"Dự đoán: {y_predict[i]:.4f} "
            f"| Thực tế: {y_test[i]:.4f}"
        )

if __name__ == "__main__":
    main()