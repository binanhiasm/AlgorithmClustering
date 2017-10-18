#Tran Quang Dat - 14520156

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from TH1_Kmeans2 import clusteringKmeans
from TH1_Spectral2 import clusteringSpectral
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern


#print ("Number of people: %d" % len(people.target_names))
#print ("Number of images: %d" % len(people.images))

#get data from file
data = np.load('dataFeatures.npy')
target = np.load('dataTarget.npy')

reduced_data = PCA(n_components=2).fit_transform(data)


#with Kmeans Clustering
chartKmeans = clusteringKmeans(data,10)
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartKmeans)
plt.title('Kmeans')

#with Spectral Clustering
chartSpectral = clusteringSpectral(data,10)
plt.figure(3)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartSpectral)
plt.title('Spectral')

plt.show()