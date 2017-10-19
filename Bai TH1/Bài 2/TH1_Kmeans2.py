#Tran Quang Dat - 14520156

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def clusteringKmeans(data,kclusters):
    y = KMeans(n_clusters=kclusters).fit_predict(data)
    return y