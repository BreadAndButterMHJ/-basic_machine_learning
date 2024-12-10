import numpy as np


class DBSCAN():

    def __init__(self, data, eps, min_samples):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.labels = np.full(data.shape[0], -1)
        self.cluster_id = -1

    def find_neighbors(self, point_id):
        neighbors_id = []
        for i in range(len(self.data)):
            if np.linalg.norm((self.data[point_id] - self.data[i]), ord=2) < self.eps:
                neighbors_id.append(i)
        return neighbors_id

    def expand_cluster(self, neighbors_id):
        visited = []
        while neighbors_id:
            current_id = neighbors_id.pop()
            if self.labels[current_id] == -1:
                self.labels[current_id] = self.cluster_id
            else:
                continue
            if len(self.find_neighbors(current_id)) >= self.min_samples:
                new_neighbors_id = self.find_neighbors(current_id)
                for neighbor in new_neighbors_id:
                    if (neighbor != current_id) and (neighbor not in neighbors_id):
                        neighbors_id.append(neighbor)

    def fit(self):
        for i in range(self.data.shape[0]):
            if self.labels[i] != -1:
                continue
            if len(self.find_neighbors(i)) < self.min_samples:
                continue
            else:
                self.cluster_id += 1
            neighbors = self.find_neighbors(i)
            self.expand_cluster(neighbors)


from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN as skDBSCAN

data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)

dbscan = DBSCAN(data, eps=0.5, min_samples=5)
dbscan1 = skDBSCAN(eps=0.5, min_samples=5)
dbscan.fit()
dbscan1.fit(data)
color = ['r', 'g', 'b', 'y', 'c', 'm']
plt.figure(1, figsize=(12, 8))
for idx, color in zip(np.unique(dbscan.labels), color):
    if idx == -1:
        plt.scatter(data[dbscan.labels == idx][:, 0], data[dbscan.labels == idx][:, 1], c='k', label='Noise')
    else:
        plt.scatter(data[dbscan.labels == idx][:, 0], data[dbscan.labels == idx][:, 1], c=color, label=f'Cluster {idx}')
plt.legend()
plt.title('DBSCAN')
print(
    f"是否和sklearn的DBSCAN一致: {'是一致' if np.sum(dbscan1.labels_ == dbscan.labels) / len(dbscan.labels) == 1 else '不一致'}")
plt.show()
