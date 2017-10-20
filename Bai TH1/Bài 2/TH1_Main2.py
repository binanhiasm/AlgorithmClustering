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
from sklearn import metrics
from sklearn.preprocessing import scale

#get data hand-written digit
digits = load_digits()
data = digits.data
target = digits.target
#KmeansClustering
def clusteringKmeans(data,kclusters):
    return KMeans(n_clusters=kclusters).fit_predict(data)

#SpectralClustering
def clusteringSpectral(data,numclusters):
    ndt = cosine_similarity(data)
    #spectral
    return SpectralClustering(n_clusters=numclusters, eigen_solver='arpack', affinity='precomputed').fit_predict(ndt)

#DBSCAN
def clusteringDBSCAN(data,eps,minSample):
    # DBSCAN
    return DBSCAN(eps=eps, min_samples=minSample).fit_predict(data)

#Agglomerative
def clusteringAgglomerative(data, numclusters):
    # Agglomerative
    return AgglomerativeClustering(n_clusters=numclusters).fit_predict(data)
#with Kmeans Clustering
chartKmeans = clusteringKmeans(data,7)
reduced_data = PCA(n_components=2).fit_transform(data)
rK =metrics.adjusted_mutual_info_score(target,chartKmeans)
print ("Compare 4 algorithms \n")
print rK
plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartKmeans)
plt.title('Kmeans')

#with Spectral Clustering
chartSpectral = clusteringSpectral(data,7)
rS =metrics.adjusted_mutual_info_score(target,chartSpectral)
print rS
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartSpectral)
plt.title('Spectral')

#with DBSCAN
chartDBSCAN = clusteringDBSCAN(data, 17.6, 1)
rD =metrics.adjusted_mutual_info_score(target,chartDBSCAN)
print rD
plt.figure(3)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartDBSCAN)
plt.title('DBSCAN')

#with Agglomerative
chartAgglomerative = clusteringAgglomerative(data, 7)
rA =metrics.adjusted_mutual_info_score(target,chartAgglomerative)
print rA
plt.figure(4)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartAgglomerative)
plt.title('Agglomerative')

# Visualize result

plt.show()