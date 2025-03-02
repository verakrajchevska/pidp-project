
#%%
from __future__ import division
import numpy as np
from multiprocessing import Pool, Manager
import timeit
import time
import matplotlib.pyplot as plt
import pandas as pd

class K_Means_parallel(object):
      
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores

    def assign_points_to_cluster(self, X):
        start_time = time.time()

        self.labels_ = [self._nearest(self.cluster_centers_, x) for x in X]
        # Map labels to data points
        indices=[]
        for j in range(self.n_clusters):
            cluster=[]
            for i, l in enumerate(self.labels_):
                if l==j: cluster.append(i)
            indices.append(cluster)
        X_by_cluster = [X[i] for i in indices]
        end_time = time.time()
        return X_by_cluster, end_time - start_time
    
    def initial_centroid(self, X):
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        return  X[initial]
    

    def fit(self, X):
        self.execution_times = []
        self.cluster_centers_ = self.initial_centroid(X)
        for i in range(self.max_iter):
            splitted_X=self._partition(X,self.num_cores)
            # Parallel Process for assigning points to clusters 
            with Pool(self.num_cores) as p:
                result = p.map(self.assign_points_to_cluster, splitted_X)
            # Collecting execution times
            times = [r[1] for r in result]
            self.execution_times.append(times)
            # Merge results 
            X_by_cluster=[]
            for c in range(0,self.n_clusters):
                r=[]
                for p in range(0,self.num_cores):
                    tmp=result[p][0][c].tolist()
                    r=sum([r, tmp ], [])
                X_by_cluster.append(np.array(r))
            
            new_centers=[c.sum(axis=0)/len(c) for c in X_by_cluster]
            new_centers = [np.array(arr) for arr in new_centers]
            old_centers=self.cluster_centers_
            old_centers = [np.array(arr) for arr in old_centers]
            # Check convergence
            if all([np.allclose(x, y) for x, y in zip(old_centers, new_centers)]) :
                self.number_of_iter=i
                break;
            else : 
                self.cluster_centers_ = new_centers
        self.number_of_iter=i
        return self
     
    # randomly shuffles and partitions the dataset
    def _partition ( self,list_in, n):
        temp = np.random.permutation(list_in)
        result = [temp[i::n] for i in range(n)]
        return result

    def _nearest(self, clusters, x):
        return np.argmin([self._distance(x, c) for c in clusters])

    def _distance(self, a, b):
        return np.sqrt(((a - b)**2).sum())

    def predict(self, X):
        return self.labels_


def run_kmeans(num_cores):
    parakmeans = K_Means_parallel(n_clusters=3, max_iter=500, num_cores=num_cores)
    parakmeans.fit(X)
    return parakmeans.execution_times


import sklearn.datasets as skl
X, y = skl.make_blobs(n_samples=150000, centers=3, cluster_std=0.60, random_state=0)

execution_times_2 = run_kmeans(2)
execution_times_4 = run_kmeans(4)
execution_times_6 = run_kmeans(6)

exec_times_2_flat = [time for sublist in execution_times_2 for time in sublist]
exec_times_4_flat = [time for sublist in execution_times_4 for time in sublist]
exec_times_6_flat = [time for sublist in execution_times_6 for time in sublist]


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(exec_times_2_flat, label='2 Cores', color='red', marker='o', linestyle='--')
ax.plot(exec_times_4_flat, label='4 Cores', color='blue', marker='o', linestyle='--')
ax.plot(exec_times_6_flat, label='6 Cores', color='green', marker='o', linestyle='--')

ax.set_xlabel('Iteration')
ax.set_ylabel('Execution Time (seconds)')
ax.legend()
ax.grid(True)

plt.show()

# %%
