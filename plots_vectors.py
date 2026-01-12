#%%

import matplotlib.pyplot as plt

num_samples = list(range(20000, 400001, 20000))

serial_times = [
    5.607531309127808, 12.583468389511108, 18.870247530937196, 24.502358746528625, 
    31.84505989551544, 37.75744295120239, 44.05103559494019, 58.08493332862854, 
    58.70671861171722, 64.85983619689941, 73.65874400138856, 83.12227826118469, 
    104.1515116930008, 98.30478053092956, 113.01561760902405, 122.00383524894714, 
    140.1022866487503, 134.19948275089263, 145.82950146198272, 134.3848307132721]

parallel_times2 = [
    11.831513929367066, 18.530110549926757, 28.537549543380738, 42.078589248657224,
    47.15789585113525, 66.19084701538085, 73.2447934627533, 89.74255919456482,
    102.08152661323547, 119.23286094665528, 126.2778480052948, 134.761954498291,
    142.03520894050598, 142.7414617061615, 153.86818332672118, 186.02605953216553,
    179.3325551509857, 217.83329739570618, 211.2442846775055, 215.76818032264708
]

parallel_times4 = [
    9.863268995285035, 15.589707279205323, 22.29314022064209, 31.66091432571411,
    34.68253440856934, 36.65735673904419, 42.334257459640504, 58.04104056358337,
    60.43719329833984, 68.04528193473816, 71.71291170120239, 72.8673629283905,
    76.75870933532715, 84.19360065460205, 93.49526562690735, 107.23526492118836,
    109.99660968780518, 105.27921504974366, 119.64137229919433, 119.23868675231934
]

parallel_times6 = [
    13.47036008834839, 15.966210556030273, 21.31583113670349, 23.275780820846556,
    30.229065465927125, 36.991486120223996, 41.27166495323181, 59.28370027542114,
    57.51531801223755, 63.71656761169434, 61.68829946517944, 76.19227557182312,
    82.48819813728332, 70.52166566848754, 86.603755235672, 89.17012777328492,
    104.23761653900146, 102.82995047569275, 108.70164470672607, 124.07023897171021
]



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
plt.savefig('1new_execution_times_vectors.png')

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

colors = ['red', 'black', 'pink', 'blue', 'green', 'yellow', 'purple', 'orange', 'magenta', 'cyan', 'maroon', 'gray' , 'darkviolet', 'lime',  'darkblue', 'gold', 'brown', 'turquoise', 'indigo', 'teal' ]

plt.figure(figsize=(12, 6))

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

plt.figure(figsize=(12, 6))

for sample_size, color in zip(speedup_data.keys(), colors):
    plt.plot(cores, speedup_data[sample_size], label=sample_size, color=color, marker='o', linestyle='--')

plt.xlabel("Number of Cores")
plt.ylabel("Speedup ratio")
plt.title("'Speedup Ratio vs Number of Cores")
plt.xticks(cores) 
plt.grid(True)
plt.legend(title="Sample Size", loc="upper right")

plt.savefig('new_speedup_ratio_vectors.png')


# STANDARD DEVIATION 

serial_std = [
    1.73579275816028, 1.5733909580738172, 3.9148045480402933, 7.007347841915894, 3.611325824645847,
    5.311799408864379, 7.528745102185198, 12.612890922578064, 6.389622467287949, 11.139867448269245,
    18.223360225955616, 14.741602946715819, 20.40769132385745, 23.736366650961287, 17.62246607432707,
    20.434941682107418, 23.609391578370083, 20.188369432912666, 32.203661023281605, 33.17589971699843
]

parallel_std2 = [
    1.1434304485908104, 2.524253571773661, 4.629278178715665, 8.929922660824403, 2.0954969464779127,
    8.548452140818515, 11.055945773740621, 16.507203340786486, 14.573968081263567, 10.598397561410478,
    11.842661783781892, 15.872291036087054, 16.422711645115353, 19.068363732886663, 16.099458939763462,
    18.485895899343195, 30.373482723298427, 11.632122191085573, 17.175608270023375, 23.19988063775493
]

parallel_std4 = [
    1.322712878224731, 2.828595608068253, 1.778617226048316, 1.9544148248289588, 6.402156949867595,
    6.3082556414469115, 7.1407806890325345, 3.823910734771505, 11.28215842879015, 5.787370955878898,
    10.45483529356258, 10.605166962939883, 10.107247237733926, 8.330721618782613, 17.80909459020303,
    13.299342582578705, 13.122329938593078, 14.326975619644639, 14.241574554059866, 15.82489384459312
]

parallel_std6 = [
    3.2859018089238354, 3.300312202759963, 2.7484518265158484, 4.2912666148181025, 6.000312353498098,
    4.88275790283918, 2.2751372638286096, 5.121459537450531, 8.935732523018059, 8.105539370523982,
    10.457498673565182, 7.801301253070284, 7.892224895517364, 8.186696685048101, 10.824418956008708,
    13.410255506204235, 3.834545427468142, 14.886413286601869, 9.940634661789698, 12.801696283760975
]


# Plotting std devs
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
