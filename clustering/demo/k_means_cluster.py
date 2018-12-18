import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class KMeans:

    def __init__(self, k, iter):
        self.k = k
        self.iter = iter
        self.marker = ['.', 'o']
        self.color = ['blanchedalmond', 'lightsalmon']

    def predict(self, feat):
        m = len(feat)
        n = len(feat[0])
        labels = np.zeros((m,), np.int)
        centering = {}
        # 初始化均值向量，随机从样本中抽取k个
        samples = random.sample(range(m+1),self.k)
        for i in range(self.k):
            centering[i] = feat[samples[i]]

        iter = 0
        # 一次迭代，计算每个样本到每个clustering的距离，将label设为距离最近的分类
        while iter < self.iter:
            for i in range(m):
                minDist = [float('Inf'), -1]    # 维持一个数组，记录最小距离以及产生的聚类
                for cluster in range(self.k):
                    dist = 0.0
                    for j in range(n):
                        dist += (centering[cluster][j] - feat[i][j]) * (centering[cluster][j] - feat[i][j])
                    dist = math.sqrt(dist)
                    if dist < minDist[0]:
                        minDist[0], minDist[1] = dist, cluster
                labels[i] = minDist[1]
            # 计算每一个分类的特征均值，赋予新的centering
            for cluster in range(self.k):
                X = np.array([feat[i] for i in range(m) if labels[i] == cluster])
                centering[cluster] = np.mean(X, axis=0)

            iter += 1

        return labels

    def plot(self, feat):
        labels = self.predict(feat)
        for cluster in range(self.k):
            X = np.array([feat[i] for i in range(len(feat)) if labels[i] == cluster])
            plt.scatter(X[:,0], X[:,1], s=10)
        plt.show()


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    clt = KMeans(5, 50)
    clt.plot(X)