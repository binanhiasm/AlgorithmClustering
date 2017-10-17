#Tran Quang Dat - 14520156

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from TH1_Kmeans2 import clusteringKmeans
from TH1_Spectral2 import clusteringSpectral

#get data hand-written digit
digits = load_digits()
data = digits.data
reduced_data = PCA(n_components=2).fit_transform(data)


#with Kmeans Clustering
chartKmeans = clusteringKmeans(data,10)
plt.figure(2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartKmeans)
plt.title('Kmeans Result')

#with Spectral Clustering
chartSpectral = clusteringSpectral(data,10)
plt.figure(3)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=chartSpectral)
plt.title('Spectral Result')



# Visualize result
plt.figure(1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=digits.target)
plt.title('Target Result')

plt.show()