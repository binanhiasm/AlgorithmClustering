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
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern


#extract feature to File
people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data = np.array([]).reshape(0, 1850)
for image in people.images:
    feature = local_binary_pattern(image, P=8, R=0.5).flatten()
    data = np.append(data,[feature],axis=0)
#save file
np.save(file='dataTarget.npy', arr=people.target)
np.save(file='dataFeatures.npy', arr=data)
