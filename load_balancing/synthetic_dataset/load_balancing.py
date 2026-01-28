
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
        self.convergence_iter = None
        for iteration in range(self.max_iter):
            
            data_chunks = np.array_split(np.random.permutation(data), self.num_cores)
            with Pool(self.num_cores) as pool:
                results = pool.map(self.assign_clusters, data_chunks)
            
            times = [r[1] for r in results]
            self.execution_times.append(times)
            
            clusters_by_chunk = [r[0] for r in results]
            clusters = self.merge_clusters(clusters_by_chunk)
            
            new_centroids = self.compute_centroids(clusters, data)

            if self.convergence_iter is None and np.allclose(self.centroids, new_centroids):
                self.convergence_iter = iteration + 1
            
            self.centroids = new_centroids
        
        self.iterations = self.max_iter
        return self

    def merge_clusters(self, clusters_by_chunk):
        merged_clusters = [[] for _ in range(self.n_clusters)]
        for cluster_set in clusters_by_chunk:
            for idx, cluster in enumerate(cluster_set):
                merged_clusters[idx].extend(cluster)
        return merged_clusters

    def predict(self, data):
        return [self.closest_centroid(point) for point in data]



import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)


def run_kmeans(num_cores, X):
    parakmeans = ParallelKMeans_LoadBalancing(n_clusters=3, max_iter=100, num_cores=num_cores)
    parakmeans.fit(X)
    return parakmeans.execution_times


def run_kmeans_multiple_times(num_cores, X, runs=10):
    all_max_execution_times = []
    all_per_core_times = []  
    
    for _ in range(runs):
        execution_times_in_iterations = run_kmeans(num_cores, X)  
        max_core_times_per_iteration = [max(core_times_in_iteration) for core_times_in_iteration in execution_times_in_iterations]  
        all_max_execution_times.append(max_core_times_per_iteration)
        all_per_core_times.append(execution_times_in_iterations)  
    
    # Computing the average maximum execution time for each iteration across runs
    averaged_max_times = np.mean(all_max_execution_times, axis=0)

    # Computing the average per-core execution time for each iteration across runs
    per_core_mean_times = np.mean(all_per_core_times, axis=0) 

    return averaged_max_times, per_core_mean_times


core_configs = [2, 4, 6]  
averaged_times_per_config = []
core_means_per_config = []

for cores in core_configs:
    averaged_max_times, per_core_mean_times = run_kmeans_multiple_times(cores, X, runs=10)
    averaged_times_per_config.append(averaged_max_times)
    core_means_per_config.append(per_core_mean_times)

    print(f'Average execution times of the maximum core time per iteration for {cores} cores configuration:')
    print(averaged_max_times)
    print(f'Average execution times per core per iteration for {cores} cores configuration: ')
    print(per_core_mean_times)

print(f'Average execution times per iteration for each core configuration: {averaged_times_per_config}')
print(f'Average execution times per core per iteartion for each core condiguration: {core_means_per_config}')

"""
np.savez(
    "load_balancing_results.npz",
    core_configs=np.array(core_configs),
    averaged_times_per_config=np.array(averaged_times_per_config, dtype=object),
    core_means_per_config=np.array(core_means_per_config, dtype=object)
)

print("Results saved to load_balancing_results.npz")


"""


# Plotting the averaged maximum execution times for each iteration across runs and core configurations
def plot_execution_times(core_configs, averaged_times_per_config):
    plt.figure(figsize=(10, 6))
    for i, times in enumerate(averaged_times_per_config):
        plt.plot(range(1, len(times) + 1), times, label=f"{core_configs[i]} cores")
    plt.title('Averaged Execution Time Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Execution Time (seconds)')
    plt.legend()
    plt.grid(True)


# Plotting the average per-core execution time for each iteration across runs and core configurations
def plot_per_core_means(core_configs, core_means_per_config):
    plt.figure(figsize=(10, 6))
    for i, core_means in enumerate(core_means_per_config):
        num_iters, num_cores = core_means.shape
        for core in range(num_cores):
            plt.plot(range(1, num_iters + 1), core_means[:, core], label=f"Core {core + 1} ({core_configs[i]} cores)")
    plt.title('Per-Core Average Execution Time Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Per-Core Average Execution Time (seconds)')
    plt.legend()
    plt.grid(True)



plot_execution_times(core_configs, averaged_times_per_config)
plt.savefig('load_balancing_averaged_exectime.png') 

plot_per_core_means(core_configs, core_means_per_config)
plt.savefig('core_means_per_iteration.png')




# %%
