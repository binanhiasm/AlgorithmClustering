#Tran Quang Dat - 14520156

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale

#get data hand-written digit
digits = load_digits()
data = digits.data
#KmeansClustering
def clusteringKmeans(data,kclusters):
    y = KMeans(n_clusters=kclusters).fit_predict(data)
    return y

#SpectralClustering
def clusteringSpectral(data,numclusters):
    ndt = cosine_similarity(data)
    #spectral
    y = SpectralClustering(n_clusters=numclusters, eigen_solver='arpack', affinity='precomputed').fit_predict(ndt)
    return y

#DBSCAN
def clusteringDBSCAN(data,eps,minSample):
    # DBSCAN
    y = DBSCAN(eps=eps, min_samples=minSample).fit_predict(data)
    return y

#Agglomerative
def clusteringAgglomerative(data, numclusters):
    # Agglomerative
    y = AgglomerativeClustering(n_clusters=numclusters).fit_predict(data)
    return y
#with Kmeans Clustering
chartKmeans = clusteringKmeans(data,7)
reduced_data = PCA(n_components=2).fit_transform(data)
plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartKmeans)
plt.title('Kmeans')

#with Spectral Clustering
chartSpectral = clusteringSpectral(data,7)
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartSpectral)
plt.title('Spectral')

#with DBSCAN
chartDBSCAN = clusteringDBSCAN(data, 17.6, 1)
plt.figure(3)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartDBSCAN)
plt.title('DBSCAN')

#with Agglomerative
chartAgglomerative = clusteringAgglomerative(data, 7)
plt.figure(4)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartAgglomerative)
plt.title('Agglomerative')

# Visualize result

plt.show()