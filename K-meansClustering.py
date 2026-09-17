import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering:
    # Khởi tạo mô hình K-Means
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    # Chọn ngẫu nhiên k điểm làm centroid ban đầu
    def randomCentroids(self, X):
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], self.k ,replace=False)
        return X[indices]

    # Tính khoảng cách Euclidean từ mỗi điểm đến các centroid
    def distanceEuclid(self, X, centroids):
        distances = []

        for point in X:
            point_distances = []

            for centroid in centroids:
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                point_distances.append(distance)

            distances.append(point_distances)
        return np.array(distances)

    # Gán mỗi điểm vào cluster có centroid gần nhất
    def clusters(self, distances):
        labels = np.argmin(distances, axis=1)
        return labels

    # Cập nhật centroid bằng trung bình các điểm trong cluster
    def updateCentroids(self, X, labels):
        new_centroids = []

        for k in range(self.k):
            cluster_points = X[labels == k]

            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[k])
            else:
                centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(centroid)

        return np.array(new_centroids)

    # Huấn luyện mô hình bằng cách lặp lại quá trình
    # gán cluster và cập nhật centroid
    def fit(self, X):
        self.centroids = self.randomCentroids(X)

        for i in range(self.max_iter):
            distances = self.distanceEuclid(X, self.centroids)
            labels = self.clusters(distances)
            self.centroids = self.updateCentroids(X, labels)
            self.labels = labels

        return self


    # Dự đoán cluster cho dữ liệu mới dựa trên centroid đã học
    def predict(self, X):
        distances = self.distanceEuclid(X, self.centroids)
        return self.clusters(distances)

    # Huấn luyện mô hình và trả về labels của dữ liệu
    def fitPredict(self, X):
        self.fit(X)
        return self.labels


def main():
    X, y = make_blobs(n_samples=400, centers=3, n_features=2, cluster_std=1.0, random_state=42)
    model = KMeansClustering(k=3, max_iter=100)
    labels = model.fitPredict(X)

    print("Labels:")
    print(labels[:20])

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker="o", s=100)
    plt.title("K-means")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    main()