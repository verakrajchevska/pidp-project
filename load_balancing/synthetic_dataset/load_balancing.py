
#%%
from __future__ import division
import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import pandas as pd


class ParallelKMeans_LoadBalancing:
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores
        self.execution_times = []

    def initialize_centroids(self, data):
        random_indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        return data[random_indices]

    def assign_clusters(self, data_chunk):
        start_time = time.time()
        cluster_assignments = [self.closest_centroid(point) for point in data_chunk]
        
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, assignment in enumerate(cluster_assignments):
            clusters[assignment].append(data_chunk[idx])
        
        end_time = time.time()
        return clusters, end_time - start_time
    
    def closest_centroid(self, point):
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def compute_centroids(self, clusters, data):
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(data[np.random.randint(data.shape[0])])
        return new_centroids

    def fit(self, data):
        self.centroids = self.initialize_centroids(data)
        for iteration in range(self.max_iter):
            
            data_chunks = np.array_split(np.random.permutation(data), self.num_cores)
            with Pool(self.num_cores) as pool:
                results = pool.map(self.assign_clusters, data_chunks)
            
            times = [r[1] for r in results]
            self.execution_times.append(times)
            
            clusters_by_chunk = [r[0] for r in results]
            clusters = self.merge_clusters(clusters_by_chunk)
            
            new_centroids = self.compute_centroids(clusters, data)
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


def run_kmeans(num_cores):
    parakmeans = ParallelKMeans_LoadBalancing(n_clusters=3, max_iter=500, num_cores=num_cores)
    parakmeans.fit(X)
    return parakmeans.execution_times


import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)

execution_times_2 = run_kmeans(2)
execution_times_4 = run_kmeans(4)
execution_times_6 = run_kmeans(6)

exec_times_2_flat = [time for sublist in execution_times_2 for time in sublist]
exec_times_4_flat = [time for sublist in execution_times_4 for time in sublist]
exec_times_6_flat = [time for sublist in execution_times_6 for time in sublist]


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(exec_times_2_flat, label='2 Cores', color='red', marker='o', linestyle='--')
ax.plot(exec_times_4_flat, label='4 Cores', color='blue', marker='o', linestyle='--')
ax.plot(exec_times_6_flat, label='6 Cores', color='green', marker='o', linestyle='--')

ax.set_xlabel('Iteration')
ax.set_ylabel('Execution Time (seconds)')
ax.legend()
ax.grid(True)

plt.show()

# %%
