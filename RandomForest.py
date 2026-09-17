import numpy as np
from DecisionTree import DecisionTree
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, NumberOfTrees=100, max_depth=4, max_features=None):
        # Số lượng cây, số feature và độ sâu tối đa của mỗi cây

        self.NumberOfTrees = NumberOfTrees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        # Train các DecisionTree trên các bootstrap samples

        self.trees = []
        NumberOfSamples = X.shape[0]

        for _ in range(self.NumberOfTrees):

            # Random sampling
            indices = np.random.choice(NumberOfSamples, NumberOfSamples, replace=True)

            X_sample = X[indices]
            y_sample = y[indices]

            # Tạo và train 1 cây
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)

            # Save cây vào rừng
            self.trees.append(tree)

        return self

    def predict(self, X):

        # Mỗi cây đưa ra một prediction
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)

        # Majority voting
        finalPredictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            finalPredictions.append(np.bincount(votes).argmax())

        return np.array(finalPredictions)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForest(NumberOfTrees=20, max_depth=3, max_features=2)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    acc = np.mean(y_predict == y_test)
    print(f"Độ chính xác (Accuracy): {acc * 100:.2f}%\n")

    results = pd.DataFrame({
        'Thực tế (True)': y_test,
        'Dự đoán (Predict)': y_predict,
        'Chính xác?': y_test == y_predict
    })
    print(results)

main()



