import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

file_path = '../Documents/Parallel_K_Means/Glove datasets/glove.6B/glove.6B.50d.txt'

words = []
vectors = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]

        vector = list(map(float, parts[1:]))
        words.append(word)
        vectors.append(vector)

X = np.array(vectors)  # Shape: (400001, 50)

sample_size = 250000  
X = X[np.random.choice(X.shape[0], size=sample_size, replace=False)]

class K_Means_parallel(object):  
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores

    def assign_points_to_cluster(self, X):
        start_time = time.time()

        self.labels_ = [self._nearest(self.cluster_centers_, x) for x in X]
        # Map labels to data points
        indices=[]
        for j in range(self.n_clusters):
            cluster=[]
            for i, l in enumerate(self.labels_):
                if l==j: cluster.append(i)
            indices.append(cluster)
        X_by_cluster = [X[i] for i in indices]
        end_time = time.time()
        return X_by_cluster, end_time - start_time
    
    def initial_centroid(self, X):
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        return  X[initial]
    
    
    def fit(self, X):
        self.execution_times = []
        self.cluster_centers_ = self.initial_centroid(X)
        for i in range(self.max_iter):
            splitted_X=self._partition(X,self.num_cores)
            # Parallel Process for assigning points to clusters 
            with Pool(self.num_cores) as p:
                result = p.map(self.assign_points_to_cluster, splitted_X)
            # Collecting execution times
            times = [r[1] for r in result]
            self.execution_times.append(times)
            # Merge results 
            X_by_cluster=[]
            for c in range(0,self.n_clusters):
                r=[]
                for p in range(0,self.num_cores):
                    tmp=result[p][0][c].tolist()
                    r=sum([r, tmp ], [])
                X_by_cluster.append(np.array(r))
        
            new_centers=[c.sum(axis=0)/len(c) for c in X_by_cluster]
            new_centers = [np.array(arr) for arr in new_centers]
            old_centers=self.cluster_centers_
            old_centers = [np.array(arr) for arr in old_centers]
            # Check convergence     
            if all([np.allclose(x, y) for x, y in zip(old_centers, new_centers)]) :
                self.number_of_iter=i
                break;
            else : 
                self.cluster_centers_ = new_centers
        self.number_of_iter=i
        return self
     
    # randomly shuffles and partitions the dataset
    def _partition ( self,list_in, n):
        temp = np.random.permutation(list_in)
        result = [temp[i::n] for i in range(n)]
        return result

    def _nearest(self, clusters, x):
        return np.argmin([self._distance(x, c) for c in clusters])

    def _distance(self, a, b):
        return np.sqrt(((a - b)**2).sum())

    def predict(self, X):
        return self.labels_


def run_kmeans(num_cores, X):
    parakmeans = K_Means_parallel(n_clusters=3, max_iter=100, num_cores=num_cores)
    parakmeans.fit(X)
    return parakmeans.execution_times


def run_kmeans_multiple_times(num_cores, X, runs=10):
    all_execution_times = []
    all_per_core_times = []  # to store times per core for each iteration across runs
    
    for _ in range(runs):
        execution_times = run_kmeans(num_cores, X)  # execution_times is a list of lists
        max_times_per_iteration = [max(times) for times in execution_times]  # take max time per iteration
        all_execution_times.append(max_times_per_iteration)
        all_per_core_times.append(execution_times)  # Raw times per core per iteration

    # Align by padding
    max_iters = max(len(run) for run in all_execution_times)
    padded_times = [run + [np.nan] * (max_iters - len(run)) for run in all_execution_times]

    # Computing the average execution time for each iteration across runs
    averaged_times = np.nanmean(padded_times, axis=0)

    # Align per-core times by padding
    per_core_padded = []
    for run in all_per_core_times:
        padded_run = run + [[np.nan] * num_cores] * (max_iters - len(run))
        per_core_padded.append(np.array(padded_run))  
    
    # Computing the mean per core per iteration
    per_core_means = np.nanmean(np.array(per_core_padded), axis=0)  # Shape: (num_iters, num_cores)

    return averaged_times, per_core_means


core_configs = [2, 4, 6]  
average_execution_times_per_core = []
core_means_per_config = []

for cores in core_configs:
    averaged_times, per_core_means = run_kmeans_multiple_times(cores, X, runs=10)
    average_execution_times_per_core.append(averaged_times)
    core_means_per_config.append(per_core_means)


# Execution times for all cores sequentially flattened across all iteartions
plt.figure(figsize=(10, 6))

for cores, per_core_means in zip(core_configs, core_means_per_config):
    # Flatten the list of lists into a single array of execution times
    concatenated_times = np.array(per_core_means).flatten()  # Shape: (num_iterations * num_cores,)
    
    plt.plot(concatenated_times, label=f"{cores} Cores", marker='o', linestyle='-', markersize=4)

plt.xlabel("Core Execution (Flattened Iterations)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times Across Iterations and Cores")
plt.legend()
plt.grid(True)

plt.savefig('flattened_iterations_load_per_core.png')

# Plotting the averaged execution times for each core configuration
def plot_execution_times(core_configs, average_execution_times_per_core):
    plt.figure(figsize=(10, 6))
    for i, times in enumerate(average_execution_times_per_core):
        plt.plot(range(1, len(times) + 1), times, label=f"{core_configs[i]} cores")
    plt.title('Averaged Execution Time Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Execution Time (seconds)')
    plt.legend()
    plt.grid(True)

# Plotting per-core average execution times for each core configuration (for stacked line graph)
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


plot_execution_times(core_configs, average_execution_times_per_core)
plot_per_core_means(core_configs, core_means_per_config)

plt.savefig('new_load_balancing_exectime.png') 
plt.savefig('core_means_per_iteration.png')



























