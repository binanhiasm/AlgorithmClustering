#Tran Quang Dat - 14520156

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import random

#random data
rd = random.randrange(1000)
digits = load_digits()
data = digits.data
ndt = np.corrcoef(data)
#spectral
y = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(ndt)
reduced_data = PCA(n_components=2).fit_transform(data)
#visualize
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.show()