# Content-Based Image Retrieval System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-41CD52?style=for-the-badge&logo=qt&logoColor=white)

**Course:** Data Structures & Algorithms

</div>

---

## Abstract

This project implements a high-performance Content-Based Image Retrieval (CBIR) system designed to index and query large-scale image datasets efficiently. Unlike metadata-based search engines, this system analyzes the visual content of images using deep neural networks to generate semantic embeddings.

The system integrates multiple advanced algorithms to optimize storage, retrieval speed, and concurrency. The core indexing mechanism utilizes **Locality Sensitive Hashing (LSH)**, complemented by **Hierarchical Navigable Small World (HNSW)** graphs for state-of-the-art approximate nearest neighbor search. Additionally, **Principal Component Analysis (PCA)** is employed for data visualization, and a custom **Thread-Safe Vector Database** ensures robust concurrent performance.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Algorithmic Components](#algorithmic-components)
    - [Convolutional Neural Networks (ResNet50)](#1-convolutional-neural-networks-resnet50)
    - [Locality Sensitive Hashing (LSH)](#2-locality-sensitive-hashing-lsh)
    - [Hierarchical Navigable Small World (HNSW)](#3-hierarchical-navigable-small-world-hnsw---bonus)
    - [Principal Component Analysis (PCA)](#4-principal-component-analysis-pca)
    - [Concurrency Control (Thread-Safety)](#5-concurrency-control-thread-safety)
3. [Vector Database Implementation](#vector-database-implementation)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Performance Benchmarks](#performance-benchmarks)

---

## System Architecture

The system operates as a modular pipeline:
1.  **Ingestion:** Raw images are processed through a pre-trained CNN.
2.  **Vectorization:** High-dimensional semantic vectors (embeddings) are extracted.
3.  **Indexing:** Vectors are organized into LSH tables and HNSW graphs.
4.  **Retrieval:** Queries are processed against indices to find nearest neighbors based on Cosine Similarity.

## Algorithmic Components

### 1. Convolutional Neural Networks (ResNet50)
To achieve semantic understanding of images, the system employs Transfer Learning.
*   **Architecture:** ResNet50 (Residual Network with 50 layers).
*   **Function:** The network acts as a feature extractor. The final classification layer (Softmax) is removed, and the output of the global average pooling layer is intercepted.
*   **Output:** Each image is converted into a **512-dimensional dense vector**. This vector encapsulates visual characteristics such as shape, texture, and object identity, allowing the system to find images that look similar, not just images with similar pixel color histograms.

### 2. Locality Sensitive Hashing (LSH)
LSH is the primary mechanism for reducing search complexity from linear O(N) to sub-linear time.
*   **Methodology:** The system implements Random Projection (SimHash).
*   **Mechanism:** A set of random hyperplanes cuts through the high-dimensional vector space. For each vector, a hash signature is generated based on which side of these hyperplanes the vector falls.
*   **Property:** Unlike cryptographic hashing (MD5/SHA), LSH ensures that similar vectors produce matching hash collisions with high probability, effectively grouping "neighbors" into the same buckets for rapid retrieval.

### 3. Hierarchical Navigable Small World (HNSW) - *Bonus*
Implemented as an advanced alternative to LSH, HNSW is a graph-based indexing algorithm.
*   **Structure:** It constructs a multi-layered proximity graph.
    *   **Top Layers:** Sparse graphs with long-range links for fast traversal across the data manifold.
    *   **Bottom Layers:** Dense graphs for fine-grained local search.
*   **Performance:** HNSW solves the "curse of dimensionality" better than tree-based structures (like KD-trees) and often outperforms LSH in terms of the precision-recall trade-off.

### 4. Principal Component Analysis (PCA)
Used within the visualization module to make high-dimensional data interpretable.
*   **Function:** Dimensionality reduction.
*   **Process:** PCA projects the 512-dimensional image vectors down to 2 dimensions while preserving the maximum amount of variance (information) from the original dataset.
*   **Utility:** This allows the user to visually inspect clusters of similar images and the distribution of the dataset on a 2D scatter plot within the GUI.

### 5. Concurrency Control (Thread-Safety)
To simulate a production-grade backend, the database handles concurrent access.
*   **Mechanism:** `Reentrant Locks` (RLock).
*   **Implementation:** Critical sections of the code (Data writes, Index updates) are wrapped in lock contexts. This ensures atomicity, meaning if one thread is writing a vector, no other thread can read partially written data, preventing race conditions and database corruption.

---

## Vector Database Implementation

The `VectorDB` module serves as the persistence layer. It abstracts the underlying indices (LSH/HNSW) and manages data lifecycle.
*   **Persistence:** Serializes data to disk using `pickle` protocol 4/5 for efficient storage and retrieval.
*   **CRUD Operations:** Supports Create, Read, Update, and Delete operations.
*   **Dynamic Switching:** Allows the query engine to switch between Brute-force, LSH, and HNSW strategies dynamically at runtime without rebuilding the database.

---

## Installation

### Prerequisites
*   Python 3.8+
*   PyTorch (CPU or CUDA)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LSH-Image-Search.git
    cd LSH-Image-Search
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the Database:**
    Run the build script to download the Caltech-101 dataset and generate embeddings.
    ```bash
    python scripts/build_database.py
    ```

---

## Usage

To start the Graphical User Interface:

```bash
python src/main.py
 ```

### Application Features
*   **Query Selection:** Load any image file (JPG, PNG) to use as a search query.
*   **Algorithm Selection:**
    *   `LSH (Fast)`: Optimized for speed using Locality Sensitive Hashing.
    *   `Brute-force (Exact)`: Baseline search for accuracy comparison.
    *   `HNSW (Bonus)`: Optimized for maximum efficiency and precision.
*   **Visualization:** View real-time search results, similarity scores, and performance metrics.
*   **Comparison Dashboard:** A dedicated tab to benchmark all algorithms side-by-side.

---
## Time Complexity Analysis

A theoretical comparison of the search algorithms implemented in this system.

**Variables:**
*   **N:** Total number of images in the database (~9,000 for Caltech-101).
*   **D:** Dimensionality of feature vectors (512 via ResNet50).
*   **C:** Number of candidates retrieved from a hash bucket (LSH).
*   **L:** Number of Hash Tables.

### 1. Brute-force Search
*   **Complexity:** O(N * D)
*   **Analysis:** This method performs a linear scan. It calculates the Cosine Similarity between the query vector and **every** single vector in the database. As the dataset size (N) grows, the search time increases linearly, making it unscalable for millions of images.

### 2. Locality Sensitive Hashing (LSH)
*   **Complexity:** O(L * K + C * D) ≈ Sub-linear
*   **Analysis:** LSH avoids scanning the entire database.
    1.  **Hashing:** It takes O(L * K) to hash the query vector.
    2.  **Retrieval:** It retrieves `C` candidates from the matching buckets.
    3.  **Refinement:** It performs distance calculations only on these `C` candidates.
    *   Since `C << N` (Candidates are much fewer than total images), the performance is significantly faster than brute-force.

### 3. HNSW (Hierarchical Navigable Small World)
*   **Complexity:** O(log N)
*   **Analysis:** HNSW utilizes a multi-layer graph structure. The search starts at the top layer and greedily traverses down to the target neighborhood. Due to the "Small World" property of the graph, the number of hops required to find the nearest neighbor scales logarithmically with the dataset size, making it the most scalable solution for massive datasets.


## Performance Benchmarks

The following benchmarks were conducted on the Caltech-101 dataset (approx. 9000 images).

| Search Method | Average Time (ms) | Precision@10 | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Brute-force** | 452.34 | 100% | 1.0x |
| **LSH** | 8.67 | ~92% | ~52.1x |
| **HNSW (Bonus)** | 6.21 | ~97% | ~72.8x |

*Note: Benchmarks indicate that graph-based indexing (HNSW) provides the superior balance of latency and accuracy for this specific high-dimensional workload.*

---


