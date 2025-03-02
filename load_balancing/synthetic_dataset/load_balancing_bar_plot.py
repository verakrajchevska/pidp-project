#%%
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import time
import sklearn.datasets as skl

class K_Means_parallel(object):
    def __init__(self, n_clusters, max_iter, num_cores):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_cores = num_cores

    def assign_points_to_cluster(self, X):
        start_time = time.time()
        self.labels_ = [self._nearest(self.cluster_centers_, x) for x in X]
        indices = [[] for _ in range(self.n_clusters)]
        for i, l in enumerate(self.labels_):
            indices[l].append(i)
        X_by_cluster = [X[i] for i in indices]
        end_time = time.time()
        return X_by_cluster, end_time - start_time, len(X)

    def initial_centroid(self, X):
        initial = np.random.permutation(X.shape[0])[:self.n_clusters]
        return X[initial]

    def fit(self, X):
        self.execution_times = []
        self.points_processed_per_core = []
        self.cluster_centers_ = self.initial_centroid(X)
        for i in range(self.max_iter):
            splitted_X = self._partition(X, self.num_cores)
            with Pool(self.num_cores) as p:
                result = p.map(self.assign_points_to_cluster, splitted_X)
            times = [r[1] for r in result]
            self.execution_times.append(times)
            points_processed = [r[2] for r in result]
            self.points_processed_per_core.append(points_processed)
            X_by_cluster = []
            for c in range(self.n_clusters):
                cluster = []
                for p in range(self.num_cores):
                    cluster.extend(result[p][0][c].tolist())
                X_by_cluster.append(np.array(cluster))
            new_centers = [np.mean(c, axis=0) for c in X_by_cluster if len(c) > 0]
            if all([np.allclose(x, y) for x, y in zip(self.cluster_centers_, new_centers)]):
                break
            else:
                self.cluster_centers_ = new_centers
        return self

    def _partition(self, list_in, n):
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
    return parakmeans.points_processed_per_core

X, y = skl.make_blobs(n_samples=10000, centers=3, cluster_std=0.60, random_state=0)

points_processed_2 = run_kmeans(2)
points_processed_4 = run_kmeans(4)
points_processed_6 = run_kmeans(6)

avg_points_processed_2 = np.mean(points_processed_2, axis=0)
avg_points_processed_4 = np.mean(points_processed_4, axis=0)
avg_points_processed_6 = np.mean(points_processed_6, axis=0)

# Creating a bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
index_2 = np.arange(2)
index_4 = np.arange(4)
index_6 = np.arange(6)

bar1 = ax.bar(index_2, avg_points_processed_2, bar_width, label='2 Cores')
bar2 = ax.bar(index_4 + bar_width, avg_points_processed_4, bar_width, label='4 Cores')
bar3 = ax.bar(index_6 + 2 * bar_width, avg_points_processed_6, bar_width, label='6 Cores')

ax.set_xlabel('Core')
ax.set_ylabel('Average Points Processed')
ax.set_title('Average Points Processed by Each Core')
ax.legend()
ax.grid(True)

core_labels_2 = ['Core 1', 'Core 2']
core_labels_4 = ['Core 1', 'Core 2', 'Core 3', 'Core 4']
core_labels_6 = ['Core 1', 'Core 2', 'Core 3', 'Core 4', 'Core 5', 'Core 6']

ax.set_xticks(np.concatenate([index_2, index_4 + bar_width, index_6 + 2 * bar_width]))
ax.set_xticklabels(core_labels_2 + core_labels_4 + core_labels_6, rotation=45)

plt.show()

# %%
