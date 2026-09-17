import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)

def main():
    data = load_iris()

    X = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNN(k=3)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    accuracy = np.mean(y_predict == y_test)

    print(f'Accuracy: {int(accuracy*100)}%')

    results = pd.DataFrame({
        'Thực tế (True)': y_test,
        'Dự đoán (Predict)': y_predict,
        'Chính xác?': y_test == y_predict
    })
    print(results)

if __name__ == "__main__":
    main()