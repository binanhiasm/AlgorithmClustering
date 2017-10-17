#Tran Quang Dat - 14520156

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import random


def clusteringSpectral(data,numclusters):
    ndt = np.corrcoef(data)
    #spectral
    y = SpectralClustering(n_clusters=numclusters, eigen_solver='arpack', affinity='precomputed').fit_predict(ndt)
    return y
