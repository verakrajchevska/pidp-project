import numpy as np
import matplotlib.pyplot as plt

data = np.load("load_balancing_results.npz", allow_pickle=True)

core_configs = data["core_configs"]
averaged_times_per_config = data["averaged_times_per_config"]
core_means_per_config = data["core_means_per_config"]

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

