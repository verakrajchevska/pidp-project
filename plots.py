#%%

import matplotlib.pyplot as plt
import pandas as pd

labels = ['serial', 'parallel cores=2', 'parallel cores=4', 'parallel cores=6']
colors = ['blue', 'orange' ,'green', 'red']

# Plotting execution times 

num_samples = list(range(50000, 500001, 50000))

file_paths_et = ['execution_times/synthetic_dataset/sim_serial_k3.csv',
                'execution_times/synthetic_dataset/sim_k3_p2.csv',
                'execution_times/synthetic_dataset/sim_k3_p4.csv',
                'execution_times/synthetic_dataset/sim_k3_p6.csv']

fig, ax = plt.subplots(figsize=(12, 6))

for file_path, label, color in zip(file_paths_et, labels, colors):
      df = pd.read_csv(file_path, sep='\t', header=None, names=['Execution number', 'Time'])
      df = df.iloc[1:] 
      ax.plot(num_samples, df['Time'], label=label, marker='o', color=color, linewidth=2)

ax.grid(True)
ax.set_xlabel('Number of samples')
ax.set_ylabel('Computational time (seconds)')
ax.set_title("Comparison of Serial and Parallel K-Means Execution Time")
ax.legend()

plt.xticks(num_samples,labels=[str(x) for x in num_samples])
plt.tight_layout()
plt.savefig('plt_results_k_means_k3.png')


# Plotting Speedup ratio

samples = [50000, 150000, 250000, 400000, 550000]

file_paths_su = ['speedup/synthethic_dataset/speedup_50000.csv', 
              'speedup/synthethic_dataset/speedup_150000.csv', 
              'speedup/synthethic_dataset/speedup_250000.csv', 
              'speedup/synthethic_dataset/speedup_400000.csv',
              'speedup/synthethic_dataset/speedup_550000.csv']

speedup_by_cores = {
    1: [],
    2: [],
    4: [],
    6: []
}

for file_path in file_paths_su:
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Cores', 'Ratio'])
    for core in speedup_by_cores.keys():
        ratio = df[df['Cores'] == core]['Ratio'].values
        if ratio.size > 0:
            speedup_by_cores[core].append(ratio[0])
        else:
            speedup_by_cores[core].append(None)


fig, ax = plt.subplots(figsize=(10, 6))

for core, color, label in zip(speedup_by_cores.keys(), colors, labels):
    ax.plot(samples, speedup_by_cores[core], label=label, color=color, marker='o', linestyle='--')

ax.grid(True)
ax.set_xlabel('Number of samples')
ax.set_ylabel('Speedup ratio')
ax.legend()

plt.xticks(samples, labels=[str(x) for x in samples])
plt.tight_layout()
plt.savefig('speedup_ratio.png')









# %%
