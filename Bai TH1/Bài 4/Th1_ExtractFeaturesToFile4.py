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
from sklearn.preprocessing import scale
from skimage.feature import hog
from skimage import color, exposure

people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data = []
count = 0
for image in people.images:
    hist = hog(image, block_norm='L2')
    data.append(hist)
    count = count + 1
print count
np.save(file='targetHOG.npy', arr= people.target)
np.save(file='HOGFeatures.npy', arr=data)