#Tran Quang Dat - 14520156

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#random data
X, Y = make_blobs(n_samples=5000, n_features=2, centers=2, random_state=2000)
#Kmeans
cluster = KMeans(n_clusters=2, random_state=2000)
result = cluster.fit_predict(X)
centers = cluster.cluster_centers_
#visualize
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=result)
plt.scatter(centers[:,0], centers[:, 1])
plt.show()