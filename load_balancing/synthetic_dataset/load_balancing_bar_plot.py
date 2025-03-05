#%%
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import time
import sklearn.datasets as skl

class ParallelKMeans_PointsByCore:
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores
        self.execution_times = []
        self.points_processed_per_core = []

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
        return clusters, end_time - start_time, len(data_chunk)

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

            clusters_by_chunk = [r[0] for r in results]
            times = [r[1] for r in results]
            points_processed = [r[2] for r in results]

            self.execution_times.append(times)
            self.points_processed_per_core.append(points_processed)
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
    parakmeans = ParallelKMeans_PointsByCore(n_clusters=3, max_iter=500, num_cores=num_cores)
    parakmeans.fit(X)
    return parakmeans.points_processed_per_core

X, y = skl.make_blobs(n_samples=10000, centers=3, cluster_std=0.60, random_state=0)

points_processed_2 = run_kmeans(2)
points_processed_4 = run_kmeans(4)
points_processed_6 = run_kmeans(6)

avg_points_processed_2 = np.mean(points_processed_2, axis=0)
avg_points_processed_4 = np.mean(points_processed_4, axis=0)
avg_points_processed_6 = np.mean(points_processed_6, axis=0)

# Creating a bar plot for the loads of each core for each core configuration
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
index_2 = np.arange(2)
index_4 = np.arange(4)
index_6 = np.arange(6)

bar1 = ax.bar(index_2, avg_points_processed_2, bar_width, label='2 Cores')
bar2 = ax.bar(index_4 + bar_width, avg_points_processed_4, bar_width, label='4 Cores')
bar3 = ax.bar(index_6 + 2 * bar_width, avg_points_processed_6, bar_width, label='6 Cores')

ax.set_xlabel('Core')
ax.set_ylabel('Average Points Processed')
ax.set_title('Average Points Processed by Each Core')
ax.legend()
ax.grid(True)

core_labels_2 = ['Core 1', 'Core 2']
core_labels_4 = ['Core 1', 'Core 2', 'Core 3', 'Core 4']
core_labels_6 = ['Core 1', 'Core 2', 'Core 3', 'Core 4', 'Core 5', 'Core 6']

ax.set_xticks(np.concatenate([index_2, index_4 + bar_width, index_6 + 2 * bar_width]))
ax.set_xticklabels(core_labels_2 + core_labels_4 + core_labels_6, rotation=45)

plt.show()

# %%
