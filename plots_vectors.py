#%%

import matplotlib.pyplot as plt
import matplotlib


num_samples = [2000, 4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000, 176000, 208000, 256000, 304000, 352000, 400000]

serial_times = [0.6075018405914306, 1.2508371591567993, 2.011733603477478, 4.66209659576416, 8.972252225875854, 12.988133025169372, 19.21689133644104, 32.97604887485504, 45.05005786418915, 58.9005710363388, 74.70431180000305, 97.07543227672576, 136.2126736164093, 115.64596951007843, 149.86407387256622]
parallel_times2 = [2.0236177682876586, 2.60832736492157, 3.62250394821167, 4.889180612564087, 7.601040458679199, 12.0547771692276, 16.272322940826417, 27.820460844039918,33.15603933334351,48.70815486907959,62.102319383621214,66.67599835395814,91.47340440750122,95.65075054168702,109.87223522663116]
parallel_times4 = [3.032279944419861,3.081585741043091,3.7561243057250975,4.802276062965393,7.195327520370483,7.8156269788742065,12.32695939540863,15.377381896972656,23.079898071289062,26.726802587509155,32.76468374729156,46.960315346717834,49.132115077972415,58.92307856082916,58.79104278087616]
parallel_times6 = [3.286440110206604, 4.218277502059936, 4.854680514335632, 6.64032096862793,7.573333716392517,9.776570892333984,11.914310312271118,19.196191668510437,19.552495455741884,23.983519124984742,30.19785342216492,40.35036220550537,48.987099051475525,50.197532296180725,63.09821407794952]

comparison_times = {
    "serial": serial_times,
    "parallel cores=2": parallel_times2,
    "parallel cores=4": parallel_times4,
    "parallel cores=6": parallel_times6,
}
 
# #EXECUTION TIMES LOG

# plt.figure(figsize=(12, 6))

# for label, times in comparison_times.items():
#     plt.plot(num_samples, times, marker='o', label=label)

# plt.xscale('log')

# ticks = [2000, 4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000, 176000, 208000, 256000, 304000, 352000, 400000]
# plt.xticks(ticks, labels=[str(x) for x in ticks], rotation=45)

# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.xlabel("Number of samples (log scale)")
# plt.ylabel("Computational Time (seconds)")
# plt.title("Comparison of Serial and Parallel K-Means Execution Time (Logarithmic Scale)")
# plt.legend()

# plt.tight_layout()
# plt.savefig('new_execution_times_log_scale.png')


#EXECUTION TIMES

# Creating an evenly spaced index for x-axis 
even_spacing = range(len(num_samples))

plt.figure(figsize=(12, 6))

for label, times in comparison_times.items():
    plt.plot(even_spacing, times, marker='o', label=label)

plt.xticks(even_spacing, labels=[str(x) for x in num_samples], rotation=45)

plt.grid(True, linestyle="--", linewidth=0.5)
plt.xlabel("Number of samples")
plt.ylabel("Computational Time (seconds)")
plt.title("Comparison of Serial and Parallel K-Means Execution Time")
plt.legend()

plt.tight_layout() 
plt.savefig('new_execution_times_vectors.png')

#SPEEDUP TIMES

# Data: Computational times for different core counts
cores = [1, 2, 4, 6]
computational_times = {}

for i, sample_size in enumerate(num_samples):
    computational_times[str(sample_size)] = [
        serial_times[i],
        parallel_times2[i],
        parallel_times4[i],
        parallel_times6[i],
    ]

colors = ['red', 'black', 'pink', 'blue', 'green', 'yellow', 'purple', 'orange', 'magenta', 'cyan', 'maroon', 'gray' , 'darkviolet', 'lime',  'darkblue' ]

plt.figure(figsize=(10, 6))

for sample_size, color in zip(computational_times.keys(), colors):
    plt.plot(cores, computational_times[sample_size], marker='o', label=sample_size, color=color)

plt.xlabel("Number of Cores")
plt.ylabel("Computational Time (seconds)")
plt.title("Computational Time vs Number of Cores")
plt.xticks(cores)  
plt.grid(True)
plt.legend(title="Sample Size", loc="upper right")

plt.savefig("new_execution_time_speedup_graph.png")

#SPEEDUP RATIO with num cores

# Calculating speedup ratios
speedup_ratios_2 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times2)]
speedup_ratios_4 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times4)]
speedup_ratios_6 = [serial / parallel for serial, parallel in zip(serial_times, parallel_times6)]

speedup_data = {}
for i, sample_size in enumerate(num_samples):
    speedup_data[str(sample_size)] = [
        1,  # Speedup ratio for serial (baseline)
        speedup_ratios_2[i],
        speedup_ratios_4[i],
        speedup_ratios_6[i],
    ]

colors = ['red', 'black', 'pink', 'blue', 'green', 'yellow', 'purple', 'orange', 'magenta', 'cyan', 'maroon', 'gray' , 'darkviolet', 'lime',  'darkblue' ]

fig, ax = plt.subplots(figsize=(10, 6))


for sample_size, color in zip(speedup_data.keys(), colors):
    ax.plot(cores, speedup_data[sample_size], label=sample_size, color=color, marker='o', linestyle='--')


ax.set_xlabel('Number of cores')
ax.set_ylabel('Speedup ratio')
ax.legend(title='Sample Sizes', loc='upper left')
ax.grid(True)

plt.xticks(cores)
plt.title('Speedup Ratio vs Number of Cores')
plt.savefig('new_speedup_ratio_vectors.png')

# %%

# %%
