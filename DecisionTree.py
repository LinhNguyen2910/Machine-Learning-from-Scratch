import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Lưu thông tin của một node: điều kiện chia hoặc giá trị của node lá
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value

    def is_leaf(self):
        # Node lá là node có giá trị dự đoán
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=4, max_features=None):
        # Khởi tạo cây với độ sâu tối đa
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def Gini(self, y):
        # Tính độ hỗn tạp Gini của các nhãn trong một node
        NumberOfSamples = len(y)

        if NumberOfSamples == 0:
            return 0

        # Tính xác suất xuất hiện của từng class
        p = np.bincount(y) / NumberOfSamples

        return 1.0 - np.sum(p**2)

    def SplitData(self, X, y, featureIndex, threshold):
        # Chia dữ liệu thành 2 nhánh dựa trên feature và threshold
        leftMask = X[:, featureIndex] <= threshold
        rightMask = X[:, featureIndex] > threshold

        return X[leftMask], y[leftMask], X[rightMask], y[rightMask]

    def FindBestSplit(self, X, y):
        # Tìm feature và threshold tạo ra Gini Gain lớn nhất

        bestGain = -1.0
        splitIndex = None
        splitThresh = None

        currentGini = self.Gini(y)
        NumberOfSamples, NumberOfFeatures = X.shape

        # Chọn feature để xét tại node hiện tại
        if self.max_features is None:
            featureIndices = range(NumberOfFeatures)
        else:
            featureIndices = np.random.choice(
                NumberOfFeatures,
                self.max_features,
                replace=False
            )

        # Chỉ tìm split trên các feature đã chọn
        for featureIndex in featureIndices:

            # Các giá trị duy nhất được dùng làm threshold
            thresholds = np.unique(X[:, featureIndex])

            # Thử từng threshold của feature hiện tại
            for thresh in thresholds:
                X_left, y_left, X_right, y_right = self.SplitData(
                    X, y, featureIndex, thresh
                )

                # Bỏ qua phép chia nếu một nhánh không có dữ liệu
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                p_left = len(y_left) / NumberOfSamples
                p_right = len(y_right) / NumberOfSamples

                # Gini Gain = Gini trước khi chia - Gini sau khi chia
                gain = currentGini - (
                    p_left * self.Gini(y_left)
                    + p_right * self.Gini(y_right)
                )

                # Lưu lại phép chia tốt nhất
                if gain > bestGain:
                    bestGain = gain
                    splitIndex = featureIndex
                    splitThresh = thresh

        return splitIndex, splitThresh

    def buildTree(self, X, y, depth=0):
        # Đệ quy xây dựng cây từ node hiện tại

        NumberOfSamples, NumberOfFeatures = X.shape
        NumberOfLabels = len(np.unique(y))

        # Dừng chia và tạo node lá khi đạt điều kiện dừng
        if depth >= self.max_depth or NumberOfLabels == 1 or NumberOfSamples < 2:
            leafValue = np.bincount(y).argmax()
            return Node(value=leafValue)

        # Tìm phép chia tốt nhất cho node hiện tại
        featureIndex, thresh = self.FindBestSplit(X, y)

        # Không tìm được phép chia → tạo node lá
        if featureIndex is None:
            return Node(value=np.bincount(y).argmax())

        # Chia dữ liệu thành hai nhánh
        X_left, y_left, X_right, y_right = self.SplitData(
            X, y, featureIndex, thresh
        )

        # Đệ quy xây dựng cây con
        leftChild = self.buildTree(X_left, y_left, depth + 1)
        rightChild = self.buildTree(X_right, y_right, depth + 1)

        # Tạo node quyết định chứa điều kiện chia và hai node con
        return Node(feature=featureIndex, threshold=thresh, left=leftChild, right=rightChild)

    def fit(self, X, y):
        # Xây dựng cây từ dữ liệu training
        self.root = self.buildTree(X, y)
        return self

    def predict(self, X):
        # Dự đoán từng mẫu bằng cách duyệt từ root xuống leaf

        predict = []
        for row in X:
            node = self.root
            while not node.is_leaf():
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predict.append(node.value)

        return np.array(predict)


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTree(max_depth=4)
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

if __name__ == "__main__":
    main()