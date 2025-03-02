#%%

import matplotlib.pyplot as plt
import pandas as pd

file_paths = ['speedup_50000.csv', 
              'speedup_150000.csv', 
              'speedup_250000.csv', 
              'speedup_400000.csv',
              'speedup_550000.csv']
labels = ['50000', '150000', '250000', '400000', '550000']
colors = ['red', 'orange', 'purple', 'blue', 'green']

fig, ax = plt.subplots(figsize=(10, 6))
for file_path, color, label in zip(file_paths, colors, labels):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Cores', 'Time'])
    ax.plot(df['Cores'], df['Time'], label=label, color=color, marker='o', linestyle='--')

ax.set_xlabel('Number of cores')
ax.set_ylabel('Speedup ratio')
ax.legend()
ax.grid(True)

plt.xticks(ticks=[1, 2, 4, 6], labels=[1, 2, 4, 6])
plt.show()
# %%
