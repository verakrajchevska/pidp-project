
import numpy as np
import timeit
from multiprocessing import Pool



class ParallelKMeans:
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores

    def initialize_centroids(self, data):
        random_indices = np.random.choice(data.shape[0])[:self.n_clusters]
        return data[random_indices]

    def assign_clusters(self, data_chunk):
        cluster_assignments = [self.closest_centroid(point) for point in data_chunk]
        # Group points by their assigned cluster
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, assignment in enumerate(cluster_assignments):
            clusters[assignment].append(data_chunk[idx])

        return clusters 
    
    def closest_centroid(self, point):
        # returns the index of the closest centroid
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def compute_centroids(self, clusters, data):
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

            data_chunks = np.array_split(np.random.permutation(data), self.num_cores)
            # Parallel assignment of points to the nearest centroid 
            with Pool(self.num_cores) as pool:
                clusters_by_chunk = pool.map(self.assign_clusters, data_chunks)
            
            # Merge clusters from different chunks
            clusters = self.merge_clusters(clusters_by_chunk)
            new_centroids = self.compute_centroids(clusters, data)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.iterations = iteration
        return self

    def merge_clusters(self, clusters_by_chunk):
        merged_clusters = [[] for _ in range(self.n_clusters)]
        for cluster_set in clusters_by_chunk:
            for idx, cluster in enumerate(cluster_set):
                merged_clusters[idx].extend(cluster)
        return merged_clusters

    def predict(self, data):
        return [self.closest_centroid(point) for point in data]
    


TEST_CODE1 = """
kmeans = SerialKMeans(n_clusters = 3, max_iter = 500)
kmeans.fit(X)
"""
TEST_CODE2 = """
parakmeans = ParallelKMeans(n_clusters = 3, max_iter = 500, num_cores = 2)
parakmeans.fit(X)
"""
TEST_CODE4 = """
parakmeans = ParallelKMeans(n_clusters = 3, max_iter = 500, num_cores = 4)
parakmeans.fit(X)
"""
TEST_CODE6 = """
parakmeans = ParallelKMeans(n_clusters = 3, max_iter = 500, num_cores = 6)
parakmeans.fit(X)
"""

# Speedup ratio for each core configuration (the ratio in the csv results was computed manually as serial_time/simulated_time for each sample subset and each core configuration)

speedup_50000 = []
speedup_150000 = []
speedup_250000 = []
speedup_400000 = []
speedup_550000 = []

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

speedup_50000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
speedup_50000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
speedup_50000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("1")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

speedup_150000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
speedup_150000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
speedup_150000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("2")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

speedup_250000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
speedup_250000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
speedup_250000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("3")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

speedup_400000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
speedup_400000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
speedup_400000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("4")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

speedup_550000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
speedup_550000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
speedup_550000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("5")

# Speedup through execution times for each core configuration

num_samples50000 = []
num_samples150000 = []
num_samples250000 = []
num_samples400000 = []
num_samples550000 = []


SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
from serial_k_means import SerialKMeans
"""

num_samples50000.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)
num_samples50000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
num_samples50000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
num_samples50000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("1")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
from serial_k_means import SerialKMeans
"""

num_samples150000.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)
num_samples150000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
num_samples150000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
num_samples150000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("2")


SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=250000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
from serial_k_means import SerialKMeans
"""

num_samples250000.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)
num_samples250000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
num_samples250000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
num_samples250000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("3")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=400000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
from serial_k_means import SerialKMeans
"""

num_samples400000.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)
num_samples400000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
num_samples400000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
num_samples400000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("4")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=550000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
from serial_k_means import SerialKMeans
"""
num_samples550000.append(timeit.timeit(stmt=TEST_CODE1,setup=SETUP_CODE,number=70)/70)
num_samples550000.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
num_samples550000.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
num_samples550000.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("5")


# Execution times for each core configuration

sim2 = []
sim4 = []
sim6 = []


SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=50000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""
sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("1")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=100000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""
sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("2")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("3")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=200000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)
print ("4")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=250000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""
sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("5")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=300000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""
sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("6")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=350000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("7")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=400000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("8")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=450000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("9")

SETUP_CODE = """
import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=500000, centers=3, cluster_std=0.60, random_state=0)
from __main__ import ParallelKMeans
"""

sim2.append(timeit.timeit(stmt=TEST_CODE2,setup=SETUP_CODE,number=70)/70)
sim4.append(timeit.timeit(stmt=TEST_CODE4,setup=SETUP_CODE,number=70)/70)
sim6.append(timeit.timeit(stmt=TEST_CODE6,setup=SETUP_CODE,number=70)/70)

print ("10")


import pandas as pd 	

results1 = pd.DataFrame(speedup_50000) 
results1.to_csv('./speedup_50000.csv', sep='\t')
results2 = pd.DataFrame(speedup_150000) 
results2.to_csv('./speedup_150000.csv', sep='\t')
results3 = pd.DataFrame(speedup_250000) 
results3.to_csv('./speedup_250000.csv', sep='\t')
results4 = pd.DataFrame(speedup_400000) 
results4.to_csv('./speedup_400000.csv', sep='\t')
results5 = pd.DataFrame(speedup_550000) 
results5.to_csv('./speedup_550000.csv', sep='\t')

results1 = pd.DataFrame(num_samples50000) 
results1.to_csv('./num_samples50000_speedup.csv', sep='\t')
results2 = pd.DataFrame(num_samples150000) 
results2.to_csv('./num_samples150000_speedup.csv', sep='\t')
results3 = pd.DataFrame(num_samples250000) 
results3.to_csv('./num_samples250000_speedup.csv', sep='\t')
results4 = pd.DataFrame(num_samples400000) 
results4.to_csv('./num_samples400000_speedup.csv', sep='\t')
results5 = pd.DataFrame(num_samples550000) 
results5.to_csv('./num_samples550000_speedup.csv', sep='\t')

results1 = pd.DataFrame(sim2)
results1.to_csv('./sim_k3_p2.csv', sep='\t')
results2 = pd.DataFrame(sim4)
results2.to_csv('./sim_k3_p4.csv', sep='\t')
results3 = pd.DataFrame(sim6)
results3.to_csv('./sim_k3_p6.csv', sep='\t')
# # %%
