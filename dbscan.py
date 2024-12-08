import numpy as np
from fontTools.ttLib.ttVisitor import visit


class DBSCAN():

    def __init__(self, data, eps, min_samples):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.labels = np.full(data.shape[0], -1)
        self.cluster_id = 0

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

# 生成数据
data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)

# 运行DBSCAN
dbscan = DBSCAN(data, eps=0.5, min_samples=5)
dbscan.fit()

print(dbscan.labels)  # 输出每个点的簇标签
