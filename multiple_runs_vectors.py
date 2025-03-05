import numpy as np
import pandas as pd
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from parallel_k_means_new import ParallelKMeans
from serial_k_means_new import SerialKMeans

file_path = '../Documents/Parallel_K_Means/Glove datasets/glove.6B/glove.6B.50d.txt'
words, vectors = [], []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        words.append(parts[0])
        vectors.append(parts[1:])

vector_data = pd.DataFrame(vectors).astype(float).to_numpy()

# parameters
num_samples = [2000, 4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000, 176000, 208000, 256000, 304000, 352000, 400000]
n_clusters = 3
max_iter = 50
num_cores = [2, 4, 6]
num_runs = 10


sampled_vectors = {n: vector_data[np.random.choice(vector_data.shape[0], size=n, replace=False)] for n in num_samples}

def monitor_resources():
    memory = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=0.1)
    return memory, cpu

def run_serial_kmeans(kmeans_class, data, n_clusters, max_iter):
    model = kmeans_class(n_clusters=n_clusters, max_iter=max_iter)
    model.fit(data)
    return model.predict(data)

def run_parallel_kmeans(kmeans_class, data, n_clusters, max_iter, num_cores):
    model = kmeans_class(n_clusters=n_clusters, max_iter=max_iter, num_cores=num_cores)
    model.fit(data)
    return model.predict(data)


def execute_parallel_with_monitoring(kmeans_class, data, n_clusters, max_iter, num_runs, initial_cores, batch_size=2):

    all_results = []
    
    for i in range(0, num_runs, batch_size):
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(run_parallel_kmeans, kmeans_class, data, n_clusters, max_iter, initial_cores) 
                       for _ in range(batch_size)]
            
            # monitoring resources while tasks are running
            for future in as_completed(futures):
                memory, _ = monitor_resources()
                if memory > 90:  # stop to avoid crash
                    print("Critical memory usage detected. Aborting batch execution.")
                    executor.shutdown(wait=False)
                    return all_results
                
                all_results.append(future.result())
        
        # garbage collection, free up memory
        gc.collect()
        
    return all_results


serial_times = []
parallel_times = {}

for n in num_samples:
    print(f"Serial execution for {n} samples")
    serial_start = time.time()
    for _ in range(num_runs):
        run_serial_kmeans(SerialKMeans, sampled_vectors[n], n_clusters, max_iter)
    serial_times.append((time.time() - serial_start) / num_runs)

    print(f"Parallel execution for {n} samples")
    parallel_times[n] = {}
    for cores in num_cores: 
        print(f"Starting with {cores} cores")
        parallel_start = time.time()
        results = execute_parallel_with_monitoring(ParallelKMeans, sampled_vectors[n], n_clusters, max_iter, num_runs, cores)
        parallel_times[n][cores] = (time.time() - parallel_start) / num_runs


print(f"Serial Times: {serial_times}")
for n, times in parallel_times.items():
    print(f"Parallel Times for {n} samples:")
    for cores, time_taken in times.items():
        print(f"  {cores} cores: {time_taken}")

