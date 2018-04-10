from sklearn.cluster import KMeans
from numpy import *
import matplotlib.pyplot as plt


X = array([[0, 1], [0, 3], [0, 4],
           [1, 0], [1, 2], [1, 4],
           [2, 1], [2, 2], [2, 4],
           [3, 0], [3, 2], [3, 3],
           [4, 0], [4, 1], [4, 3]])
Y = array([[0, 0], [1, 1], [4, 4]])
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
centers = kmeans.cluster_centers_
c_colors = zeros([1, k])
for i in range(k):
    c_colors[0, i] = i

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=75, c=kmeans.labels_, marker="o")
# plt.scatter(Y[:, 0], Y[:, 1], s=75, c=kmeans.predict(Y), marker="x")
plt.scatter(centers[:, 0], centers[:, 1], s=75, c=c_colors[0], marker="*")
plt.show()

# print(centers)
print(kmeans.labels_)
print(kmeans.predict(Y))
