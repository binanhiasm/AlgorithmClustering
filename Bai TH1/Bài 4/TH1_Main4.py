#Tran Quang Dat - 14520156

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import cluster
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import glob
from skimage.io import imread
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import color, exposure
from sklearn.preprocessing import scale
from sklearn import metrics

#get data from file
data1 = np.load('HOGFeatures.npy')
target = np.load('dataTarget.npy')
data = scale(data1)
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
rK = metrics.adjusted_rand_score(target,chartKmeans)
print rK
plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartKmeans)
plt.title('Kmeans')

#with Spectral Clustering
chartSpectral = clusteringSpectral(data,7)
rS = metrics.adjusted_rand_score(target,chartSpectral)
print rS
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartSpectral)
plt.title('Spectral')

#with DBSCAN
chartDBSCAN = clusteringDBSCAN(data, 17.6, 1)
rD = metrics.adjusted_rand_score(target,chartDBSCAN)
print rD
plt.figure(3)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartDBSCAN)
plt.title('DBSCAN')

#with Agglomerative
chartAgglomerative = clusteringAgglomerative(data, 7)
rA = metrics.adjusted_rand_score(target,chartAgglomerative)
print rA
plt.figure(4)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartAgglomerative)
plt.title('Agglomerative')


plt.show()


