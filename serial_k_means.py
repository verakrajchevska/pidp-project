#%%

import numpy as np
import timeit


class SerialKMeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def initialize_centroids(self, data):
        random_indices = np.random.choice(data.shape[0])[:self.n_clusters]
        return data[random_indices]

    def assign_clusters(self, data):
        self.cluster_assignments = [self.closest_centroid(point) for point in data]
        # Grouping points by their assigned cluster
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, assignment in enumerate(self.cluster_assignments):
            clusters[assignment].append(data[idx])
        return clusters

    def closest_centroid(self, point):
        # Calculating distances to all centroids then return the index of the closest one
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def compute_centroids(self, clusters, data):
        # compute new centroids using the mean
        new_centroids = []
        for cluster in clusters:
            if cluster:  # if cluster is empty
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(data[np.random.randint(data.shape[0])])
                # reinitialize empty clusters
        return new_centroids


    def fit(self, data):
        self.centroids = self.initialize_centroids(data)
        for iteration in range(self.max_iter):
            # Assigning points to the nearest centroid
            clusters = self.assign_clusters(data)
            new_centroids = self.compute_centroids(clusters, data)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.iterations = iteration
        return self

    def predict(self, data):
        return [self.closest_centroid(point) for point in data]

sim1 = []

TEST_CODE1 = """
kmeans = SerialKMeans(n_clusters = 3, max_iter = 500)
kmeans.fit(X)
"""

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("1")


SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=100000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("2")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("3")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=200000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("4")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=250000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""
sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("5")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=300000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("6")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=350000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("7")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=400000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)


print ("8")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=450000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)

print ("9")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=500000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import SerialKMeans
"""

sim1.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)


print ("10")


import pandas as pd
results = pd.DataFrame(sim1)
results.to_csv('./sim_serial_k3.csv', sep='\t')

# %%
