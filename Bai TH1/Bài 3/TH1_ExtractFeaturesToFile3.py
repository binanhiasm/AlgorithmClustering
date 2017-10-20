#Tran Quang Dat - 14520156

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern

#extract feature to File
people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
data = []
#count = 0
def peopleHistogram(img):
    feature = local_binary_pattern(img, P=8, R=0.5, method='uniform')
    hist, bins = np.histogram(feature.ravel(), bins=256)
    return hist
for image in people.images:
    hist = peopleHistogram(image)
    data.append(hist)
np.save(file='dataTarget.npy', arr=people.target)
np.save(file='dataFeatures.npy', arr=data)