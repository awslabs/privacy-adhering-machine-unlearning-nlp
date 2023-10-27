import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz

num_class = 2
features = np.load("./features_new.npy")
labels = np.load("./labels_new.npy")
print(len(features))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
print(len(y_train), len(y_test))
np.save(f'sst{num_class}_train.npy', {'X': X_train, 'y': y_train})
np.save(f'sst{num_class}_test.npy', {'X': X_test, 'y': y_test})
