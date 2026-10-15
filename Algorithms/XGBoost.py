
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=4, max_features=None, lambda_param=1, gamma=0):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
        self.lambda_param = lambda_param
        self.gamma = gamma

    def LeafValue(self, gradient, hessian):
        G = np.sum(gradient)
        H = np.sum(hessian)
        return -G / (H + self.lambda_param)

    def FindBestSplit(self, X, gradient, hessian):
        bestGain = 0.0
        splitIndex = None
        splitThresh = None

        NumberOfSamples, NumberOfFeatures = X.shape

        if NumberOfSamples < 2:
            return None, None, 0.0

        G = np.sum(gradient)
        H = np.sum(hessian)

        if self.max_features is None:
            featureIndices = range(NumberOfFeatures)
        else:
            n_features = min(self.max_features, NumberOfFeatures)
            featureIndices = np.random.choice(NumberOfFeatures, n_features, replace=False)

        scoreParent = G**2 / (H + self.lambda_param)

        for featureIndex in featureIndices:
            values = np.unique(X[:, featureIndex])
            thresholds = (values[:-1] + values[1:]) / 2

            for thresh in thresholds:
                leftMask = X[:, featureIndex] <= thresh
                rightMask = ~leftMask

                if not np.any(leftMask) or not np.any(rightMask):
                    continue

                G_left = np.sum(gradient[leftMask])
                H_left = np.sum(hessian[leftMask])
                G_right = np.sum(gradient[rightMask])
                H_right = np.sum(hessian[rightMask])

                scoreLeft = G_left**2 / (H_left + self.lambda_param)
                scoreRight = G_right**2 / (H_right + self.lambda_param)

                gain = 0.5 * (scoreLeft + scoreRight - scoreParent) - self.gamma

                if gain > bestGain:
                    bestGain = gain
                    splitIndex = featureIndex
                    splitThresh = thresh

        return splitIndex, splitThresh, bestGain

    def buildTree(self, X, gradient, hessian, depth=0):
        NumberOfSamples = X.shape[0]

        if depth >= self.max_depth or NumberOfSamples < 2:
            return Node(value=self.LeafValue(gradient, hessian))

        featureIndex, thresh, gain = self.FindBestSplit(X, gradient, hessian)

        if featureIndex is None or gain <= 0:
            return Node(value=self.LeafValue(gradient, hessian))

        leftMask = X[:, featureIndex] <= thresh
        rightMask = ~leftMask

        leftChild = self.buildTree(X[leftMask], gradient[leftMask], hessian[leftMask], depth + 1)
        rightChild = self.buildTree(X[rightMask], gradient[rightMask], hessian[rightMask], depth + 1)

        return Node(feature=featureIndex, threshold=thresh, left=leftChild, right=rightChild)

    def fit(self, X, gradient, hessian):
        self.root = self.buildTree(X, gradient, hessian)
        return self

    def predict(self, X):
        predictions = []

        for row in X:
            node = self.root

            while not node.is_leaf():
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.value)

        return np.array(predictions)

class XGBoost:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3, lambda_param=1, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.trees = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        positive_rate = np.mean(y)
        positive_rate = np.clip(positive_rate, 1e-6, 1 - 1e-6)

        base_score = np.log(positive_rate / (1 - positive_rate))
        self.base_score = base_score

        raw_predictions = np.full(len(y), base_score)
        self.trees = []

        for _ in range(self.n_estimators):
            probabilities = self.sigmoid(raw_predictions)

            gradient = probabilities - y
            hessian = probabilities * (1 - probabilities)

            tree = DecisionTree(max_depth=self.max_depth, lambda_param=self.lambda_param, gamma=self.gamma)
            tree.fit(X, gradient, hessian)
            raw_predictions += (self.learning_rate * tree.predict(X))

            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        raw_predictions = np.full(len(X), self.base_score)

        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)

        return self.sigmoid(raw_predictions)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


def main():
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBoost(n_estimators=30, learning_rate=0.1, max_depth=3)

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    acc = np.mean(y_predict == y_test)
    print(f"Độ chính xác (Accuracy): {acc * 100:.2f}%\n")

    results = pd.DataFrame({
        "Thực tế (True)": y_test,
        "Dự đoán (Predict)": y_predict,
        "Xác suất lớp 1": probabilities,
        "Chính xác?": y_test == y_predict})

    print(results.head(15))


if __name__ == "__main__":
    main()
