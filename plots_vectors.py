#%%


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kmeans_timing_results_glove50d.csv")

num_samples = sorted(df["num_samples"].unique())

serial_times = (
    df[(df.method == "serial")]
    .sort_values("num_samples")["mean_time"]
    .tolist()
)

parallel_times2 = (
    df[(df.method == "parallel") & (df.cores == 2)]
    .sort_values("num_samples")["mean_time"]
    .tolist()
)

parallel_times4 = (
    df[(df.method == "parallel") & (df.cores == 4)]
    .sort_values("num_samples")["mean_time"]
    .tolist()
)

parallel_times6 = (
    df[(df.method == "parallel") & (df.cores == 6)]
    .sort_values("num_samples")["mean_time"]
    .tolist()
)

serial_std = (
    df[(df.method == "serial")]
    .sort_values("num_samples")["std_time"]
    .tolist()
)

parallel_std2 = (
    df[(df.method == "parallel") & (df.cores == 2)]
    .sort_values("num_samples")["std_time"]
    .tolist()
)

parallel_std4 = (
    df[(df.method == "parallel") & (df.cores == 4)]
    .sort_values("num_samples")["std_time"]
    .tolist()
)

parallel_std6 = (
    df[(df.method == "parallel") & (df.cores == 6)]
    .sort_values("num_samples")["std_time"]
    .tolist()
)


comparison_times = {
    "serial": serial_times,
    "parallel cores=2": parallel_times2,
    "parallel cores=4": parallel_times4,
    "parallel cores=6": parallel_times6,
}
 

#EXECUTION TIMES

plt.figure(figsize=(12, 6))

for label, times in comparison_times.items():
    plt.plot(num_samples, times, marker='o', label=label)

plt.xticks(num_samples,labels=[str(x) for x in num_samples], rotation=45)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("Number of samples")
plt.ylabel("Computational Time (seconds)")
plt.title("Comparison of Serial and Parallel K-Means Execution Time")
plt.legend()

plt.tight_layout()
plt.savefig('execution_times_vectors.png')


#SPEEDUP RATIO 

cores = [1, 2, 4, 6]
# Calculating speedup ratios
speedup_ratios_1 = [1 for _ in range(20)]
speedup_ratios_2 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times2)]
speedup_ratios_4 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times4)]
speedup_ratios_6 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times6)]

speedup_data = [speedup_ratios_1, speedup_ratios_2, speedup_ratios_4, speedup_ratios_6]
labels = ['serial', 'parallel cores=2', 'parallel cores=4', 'parallel cores=6']
colors = ['blue', 'orange' ,'green', 'red']


fig, ax = plt.subplots(figsize=(12, 6))

for ratio, color, label in zip(speedup_data, colors, labels):
    ax.plot(num_samples, ratio, label=label, color=color, marker='o', linestyle='--')

ax.grid(True)
ax.set_xlabel('Number of samples')
ax.set_ylabel('Speedup ratio')
ax.set_title('Speedup Ratio vs Number of Samples')
ax.legend()

plt.xticks(num_samples,labels=[str(x) for x in num_samples], rotation=45)
plt.tight_layout()
plt.savefig('speedup_ratio_vectors.png')




# STANDARD DEVIATION 

plt.figure(figsize=(12, 6))
plt.plot(num_samples, serial_std, label="Serial", marker='o')
plt.plot(num_samples, parallel_std2, label="Parallel (2 cores)", marker='s')
plt.plot(num_samples, parallel_std4, label="Parallel (4 cores)", marker='^')
plt.plot(num_samples, parallel_std6, label="Parallel (6 cores)", marker='d')

plt.xlabel("Sample Size")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviations for Serial and Parallel Execution")
plt.legend()
plt.grid(True)

plt.xticks(num_samples, labels=[str(x) for x in num_samples], rotation=45)
plt.tight_layout()
plt.savefig('std_devs_vectors.png')


# %%

# %%
