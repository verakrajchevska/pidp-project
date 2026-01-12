import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_balancing.synthetic_dataset.load_balancing import ParallelKMeans_LoadBalancing


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


def run_kmeans(num_cores, X):
    parakmeans = ParallelKMeans_LoadBalancing(n_clusters=3, max_iter=100, num_cores=num_cores)
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

    
    # Computing the average execution time for each iteration across runs
    averaged_times = np.mean(all_execution_times, axis=0)

    # Computing the mean per core per iteration
    per_core_means = np.mean(all_per_core_times, axis=0) 

    return averaged_times, per_core_means


core_configs = [2, 4, 6]  
average_execution_times_per_core = []
core_means_per_config = []

for cores in core_configs:
    averaged_times, per_core_means = run_kmeans_multiple_times(cores, X, runs=10)
    average_execution_times_per_core.append(averaged_times)
    core_means_per_config.append(per_core_means)


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


# Stacked line graph
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



# Execution times for all cores sequentially flattened across all iteartions
def plot_flattened_execution_times(core_configs, core_means_per_config):
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


# Plot the results
plot_execution_times(core_configs, average_execution_times_per_core)
plt.savefig('1new_load_balancing_exectime.png') 

plot_per_core_means(core_configs, core_means_per_config)
plt.savefig('1core_means_per_iteration.png')

plot_flattened_execution_times(core_configs, core_means_per_config)
plt.savefig('1flattened_iterations_load_per_core.png')























