import numpy as np

class Perceptron:
    def __init__(self, max_iters=1000):
        self.max_iters = max_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        NumberOfSamples, NumberOfFeatures = X.shape
        self.weights = np.zeros(NumberOfFeatures)
        self.bias = 0

        for _ in range(self.max_iters):
            data_misclassifed = 0
            for i in range(NumberOfSamples):
                z = np.dot(X[i], self.weights) + self.bias
                if z >= 0:
                    predict = 1
                else:
                    predict = -1
                if predict != y[i]:
                    self.weights += y[i] * X[i]
                    self.bias += y[i]
                    data_misclassifed += 1
            if data_misclassifed == 0:
                break

    def predict(self, X):
        ans = []
        for x in X:
            y = np.dot(x, self.weights) + self.bias
            if y >= 0:
                predict = 1
            else:
                predict = -1
            ans.append(predict)
        return np.array(ans)

def main():
    X_test = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [5, 5],
        [6, 5],
        [5, 6],
        [6, 6]
    ])

    y_test = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    model = Perceptron(max_iters=100)
    model.fit(X_test, y_test)

    predict = model.predict(X_test)

    print("Predictions:", predict)
    print("Real:     ", y_test)
if __name__ == "__main__":
    main()