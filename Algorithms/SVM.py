import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        y = np.where(y <= 0, -1.0, 1.0)
        for _ in range(self.n_iters):
            for idx, xi in enumerate(X):
                margin = y[idx] * (np.dot(xi, self.w) + self.b)

                if margin >= 1:
                    dw = self.lambda_param * self.w
                    db = 0.0

                else:
                    dw = (self.lambda_param * self.w - y[idx] * xi)
                    db = -y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def HingeLoss(self, X, y):
        scores = np.dot(X, self.w) + self.b
        margins = y * scores
        losses = np.maximum(0, 1 - margins)
        losses = np.mean(losses)
        return losses

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b

        return np.where(linear_output >= 0, 1, -1)

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.where(y <= 0, -1, 1)
        accuracy = np.mean(y_pred == y)

        return accuracy

def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 0, 1)

    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        zero_division=0
    ))
if __name__ == "__main__":
    main()