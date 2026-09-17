import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, max_iters=1000, learningRate=0.001):
        self.max_iters = max_iters
        self.lr = learningRate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        NumberOfSamples, NumberOfFeatures = X.shape
        self.weights = np.zeros(NumberOfFeatures)
        self.bias = 0

        for _ in range(self.max_iters):
            y_predict = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / NumberOfSamples) * np.dot(X.T, (y_predict - y))
            db = (1 / NumberOfSamples) * np.sum(y_predict - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        prob = sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict = (prob >= 0.5).astype(int)
        return y_predict

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iters=1000, learningRate=0.001)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = np.mean(y_predict == y_test)

    print(f'Accuracy: {accuracy*100:.3f}%')
    print("Dự đoán: ", y_predict[:10])
    print("Thực tế:  ", y_test[:10])
if __name__ == "__main__":
    main()

