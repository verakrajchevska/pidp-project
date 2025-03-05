#%%

# Plotting the Speedup ratio

import matplotlib.pyplot as plt
import pandas as pd

file_paths_speedup = ['speedup_50000.csv', 
              'speedup_150000.csv', 
              'speedup_250000.csv', 
              'speedup_400000.csv',
              'speedup_550000.csv']

labels = ['50000', '150000', '250000', '400000', '550000']
colors = ['red', 'orange', 'purple', 'blue', 'green']
cores = [1, 2 , 4, 6]

fig, ax = plt.subplots(figsize=(10, 6))
for file_path, color, label in zip(file_paths_speedup, colors, labels):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Cores', 'Time'])
    ax.plot(df['Cores'], df['Time'], label=label, color=color, marker='o', linestyle='--')

ax.set_xlabel('Number of cores')
ax.set_ylabel('Speedup ratio')
ax.legend()
ax.grid(True)

plt.xticks(ticks=cores, labels=cores)
plt.show()
plt.savefig("speedup_ratio.png")


# Plotting the Speedup Times 

file_paths_speedup_times = ['num_samples50000_speedup.csv', 
              'num_samples150000_speedup.csv', 
              'num_samples250000_speedup.csv', 
              'num_samples400000_speedup.csv',
              'num_samples550000_speedup.csv']


fig, ax = plt.subplots(figsize=(10, 6))
for file_path, color, label in zip(file_paths_speedup_times, colors, labels):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Cores', 'Time'])
    ax.plot(df['Cores'], df['Time'], label=label, color=color, marker='o')

plt.xlabel("Number of Cores")
plt.ylabel("Computational Time (seconds)") 
plt.grid(True)
plt.legend()

plt.xticks(ticks=cores, labels=cores)
plt.show()
plt.savefig("results_speedup.png")

# Plotting the Execution times

file_paths_times = ['sim_serial_k3.csv', 
              'sim_k3_p2.csv', 
              'sim_k3_p4.csv', 
              'sim_k3_p6.csv']

labels_extime = ['1000', '10000' ,'100000', '150000', '200000','250000','300000' ,'350000', '400000', '500000']
colors_extime = ['red', 'purple', 'blue', 'green']
legends = ['serial', 'parallel cores=2', 'parallel cores=4', 'parallel cores=6']

fig, ax = plt.subplots(figsize=(10, 6))
for file_path, color, legend in zip(file_paths_times, colors_extime, legends):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Index', 'Time'])
    ax.plot(labels_extime, df['Time'], label=legend, color=color, marker='o')

plt.xlabel("Number of samples")
plt.ylabel("Computational Time (seconds)")
plt.grid(True)
plt.legend()

plt.xticks(ticks=range(len(labels_extime)), labels=labels_extime)
plt.show()
plt.savefig("results_k_means_k3.png")





# %%
