# Parallelization of K-Means Algorithm (In progress)

## Overview

This project is part of the course [Parallel and Distributed Processing @FCSE Skopje](https://finki.ukim.mk/mk/subject/%D0%BF%D0%B0%D1%80%D0%B0%D0%BB%D0%B5%D0%BB%D0%BD%D0%BE-%D0%B8-%D0%B4%D0%B8%D1%81%D1%82%D1%80%D0%B8%D0%B1%D1%83%D0%B8%D1%80%D0%B0%D0%BD%D0%BE-%D0%BF%D1%80%D0%BE%D1%86%D0%B5%D1%81%D0%B8%D1%80%D0%B0%D1%9A%D0%B5-0). It is an ongoing project with work in progress. The initial work, which was graded for the course, is consisted in the paper under research/Parallel_K_Means_Clustering_Algorithm.pdf and it serves as a baseline from which I am building on this project in consultation with course instructors.

K-means is one of the most commonly used algorithms for clustering and is well-known for being efficient, straightforward and easy to implement. Despite how easy it is to understand, the serial implementation of the K-Means algorithm has several drawbacks, especially when dealing with large datasets or requiring high performance. In a modern multi-core system the serial implementation is limited on a single CPU core, which restricts its scalability and makes processing large datasets inefficient and time-consuming, leading to underutilization of available computational resources and imbalances in resource usage. The computational time in a serial K-Means algorithm increases linearly with the number of data points and the number of clusters because each iteration requires computing the distance between each data point and all cluster centroids, which can be computationally intensive. That’s why this version is
unsuitable for real-time processing, high-dimensional data and big data applications. Also memory access is confined to a single core, which can become a bottleneck when handling large datasets, leading to slower memory access times and increased latency.

K-means is highly parallelizable, allowing the task of assigning points to clusters to be easily distributed across multiple CPUs, with each distance calculation occurring independently. The parallel implementation of the K-Means algorithm is often preferred, as it can leverage multiple cores to improve performance and efficiency. By splitting the task across multiple CPUs, it drastically cuts down the computation time, has improved scalability, enhanced resource utilization and increased throughput.

The solution architecture explains how we can achieve the task parallelization of the K-Means algorithm by incorporating the master-worker model of parallel computing architecture. In this project, I am carrying out experiments to assess how my parallel approach compares with the original K-means algorithm, specifically examining execution times and efficiency across different CPU setups.

## Project Setup

This project has a minimal set up consisting only of cloning the repository and seting up the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The generated graphs present in the initial paper can vary greatly based on the hardware configuration on which this project is run.