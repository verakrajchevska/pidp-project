import numpy as np
import pandas as pd
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from parallel_k_means import ParallelKMeans
from serial_k_means import SerialKMeans

file_path = '../Documents/Parallel_K_Means/Glove datasets/glove.6B/glove.6B.50d.txt'
words, vectors = [], []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        words.append(parts[0])
        vectors.append(parts[1:])

vector_data = pd.DataFrame(vectors).astype(float).to_numpy()

# parameters
num_samples = list(range(20000, 400001, 20000))
n_clusters = 3
max_iter = 50
num_cores = [2, 4, 6]
num_runs = 10


sampled_vectors = {n: vector_data[np.random.choice(vector_data.shape[0], size=n, replace=False)] for n in num_samples}

def run_serial_kmeans(kmeans_class, data, n_clusters, max_iter):
    model = kmeans_class(n_clusters=n_clusters, max_iter=max_iter)
    model.fit(data)
    return model.predict(data)

def run_parallel_kmeans(kmeans_class, data, n_clusters, max_iter, num_cores):
    model = kmeans_class(n_clusters=n_clusters, max_iter=max_iter, num_cores=num_cores)
    model.fit(data)
    return model.predict(data)


def multiple_runs_parallel(kmeans_class, data, n_clusters, max_iter, num_runs, initial_cores, batch_size=2):
    all_results = []
    parallel_run_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        result = run_parallel_kmeans(kmeans_class, data, n_clusters, max_iter, initial_cores)

        all_results.append(result)
        parallel_run_times.append(time.time() - start_time)

    return all_results, parallel_run_times

serial_times = []
serial_std_devs = []
parallel_times = {}
parallel_std_devs = {}

for n in num_samples:
    print(f"Serial execution for {n} samples")
    serial_run_times = [] 
    for _ in range(num_runs):
        start_time = time.time()
        run_serial_kmeans(SerialKMeans, sampled_vectors[n], n_clusters, max_iter)
        serial_run_times.append(time.time() - start_time)
    
    serial_times.append(np.mean(serial_run_times))  
    serial_std_devs.append(np.std(serial_run_times))  

    print(f"Parallel execution for {n} samples")
    parallel_times[n] = {}
    parallel_std_devs[n] = {}
    for cores in num_cores: 
        print(f"Starting with {cores} cores")
        results, parallel_run_times = multiple_runs_parallel(ParallelKMeans, sampled_vectors[n], n_clusters, max_iter, num_runs, cores)
        
        parallel_times[n][cores] = np.mean(parallel_run_times) 
        parallel_std_devs[n][cores] = np.std(parallel_run_times) 


print(f"Serial Average Times: {serial_times}")
for n, times in parallel_times.items():
    print(f"Parallel Average Times for {n} samples:")
    for cores, time_taken in times.items():
        print(f"  {cores} cores: {time_taken}")

print(f"Serial Standard Deviations: {serial_std_devs}")
for n, std_devs in parallel_std_devs.items():
    print(f"Parallel Standard Deviations for {n} samples:")
    for cores, sd in std_devs.items():
        print(f"  {cores} cores: {sd}")



rows = []

for i, n in enumerate(num_samples):
    # serial
    rows.append({
        "num_samples": n,
        "method": "serial",
        "cores": 1,
        "mean_time": serial_times[i],
        "std_time": serial_std_devs[i]
    })

    # parallel
    for cores in num_cores:
        rows.append({
            "num_samples": n,
            "method": "parallel",
            "cores": cores,
            "mean_time": parallel_times[n][cores],
            "std_time": parallel_std_devs[n][cores]
        })

df = pd.DataFrame(rows)

df.to_csv("kmeans_timing_results_glove50d.csv", index=False)


