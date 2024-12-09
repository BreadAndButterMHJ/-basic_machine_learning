import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X - self.mean_, axis=0)

    def transform(self, X):
        resX = (X - self.mean_) / self.scale_
        return resX


class PCA():
    def __init__(self, n_components, center=True):
        self.n_components = n_components
        self.center = center
        self.components = None

    def fit(self, X):
        X = X - np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)
        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]
        self.components = eig_vec[:, :self.n_components]

    def transform(self, X):
        if self.center:
            X = X - np.mean(X, axis=0)
        return np.dot(X, self.components)


# 验证结果

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.decomposition import PCA as PCA_sklearn

    data = datasets.load_iris()
    X = data.data
    pca = PCA(n_components=2)
    pca.fit(X)
    pca2 = PCA_sklearn(n_components=2)
    pca2.fit(X)
    print(pca.components, pca2.components_, sep='\n')
