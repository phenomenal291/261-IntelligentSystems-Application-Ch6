# Spotify Song Recommendation System using Unsupervised Clustering

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Spotify API Theme](https://img.shields.io/badge/UI-Spotify%20Dark-1DB954.svg)](https://developer.spotify.com/)

An interactive, high-performance web application demonstrating **Unsupervised Clustering Algorithms (K-Means, Agglomerative Hierarchical, and DBSCAN)** and a **Multi-Dimensional Audio Feature Similarity Engine** inspired by Spotify's *Discover Weekly* recommendation system.

Developed as the official interactive application for **Chapter 6: Clustering Algorithms & Cluster Validation** in **HK261 Intelligent Systems (CO3061)** at **Ho Chi Minh City University of Technology (HCMUT)**.

---

## 📌 Project Overview

1. **Curated Real-World Music Dataset (100 Iconic Tracks)**:
   - Spans diverse genres: **Pop, EDM / Progressive House, Hip-Hop / Rap, Indie / Alternative Rock, Acoustic / Melancholic Ballads, R&B / Soul, Latin Pop / Reggaeton, and Ambient / Modern Classical**.
   - Standardized 8D Spotify Audio Feature Vector:
     $$\mathbf{x} = [\text{danceability}, \text{energy}, \text{valence}, \text{tempo}, \text{acousticness}, \text{instrumentalness}, \text{speechiness}, \text{loudness}]$$
   - 2D Principal Component Analysis (PCA) projection preserving **62.08%** total variance for real-time feature space mapping.

2. **Core Clustering Algorithms Implemented (Chapter 6 Curriculum)**:
   - **K-Means Clustering**: Minimizes the within-cluster Sum of Squared Errors (SSE / Inertia). Supports `k-means++` and `random` centroid initialization.
   - **Agglomerative Hierarchical Clustering**: Builds a bottom-up dendrogram with customizable proximity links:
     - **Ward**: Minimizes within-cluster variance (balanced spherical genres).
     - **Complete Link (MAX)**: Avoids elongated chains.
     - **Average Link (MEAN)**: Balances variance and proximity.
     - **Single Link (MIN)**: Demonstrates the theoretical *chaining effect*.
   - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Discovers arbitrary density contours with parameter radius $\epsilon$ and $\text{MinPts}$. Naturally flags experimental ambient/classical tracks as **Noise / Outliers**!

3. **Interactive Visualizations & Recommendation Engine**:
   - **2D Audio Feature Space Canvas**: Interactive HTML5 scatter canvas mapping all tracks, cluster centroids ($C_k$), and ray vectors connecting the seed track to top recommendations.
   - **Audio Feature Radar (Spider) Chart**: 6-axis real-time comparative radar profile (Danceability, Energy, Valence, Acousticness, Instrumentalness, Speechiness) comparing the query track against the top recommendation.
   - **Validation Curves Modal**: Dynamic Chart.js plots for **Elbow Method (SSE vs $K$)** and **Silhouette Coefficient vs $K$** ($K \in [2, 8]$) indicating the optimal number of music moods.
   - **16 Curated 1-Click Quick Samples**: Instant seed songs including *Blinding Lights*, *Shape of You*, *Someone Like You*, *Bad Guy*, *Wake Me Up*, *HUMBLE.*, *Bohemian Rhapsody*, *Sunflower*, etc.
   - **Distance Metrics**: Cosine Similarity, Euclidean Distance ($L_2$), and Manhattan Distance ($L_1$).

---

## 📐 Mathematical Formulations

### 1. K-Means Objective Function (Sum of Squared Errors - SSE)
$$\text{SSE} = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \mathbf{m}_k\|^2$$
where $\mathbf{m}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}$ is the centroid of cluster $C_k$.

### 2. Silhouette Coefficient
For a song point $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where:
- $a(i)$ is the mean intra-cluster distance between $i$ and all other points in the same cluster.
- $b(i)$ is the minimum mean distance from $i$ to points in any other cluster.
- $s(i) \in [-1, +1]$, where values $> 0.35$ represent strong cluster cohesion and separation.

### 3. Audio Cosine Similarity for Recommendation
Given query track $\mathbf{u}$ and candidate track $\mathbf{v}$:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
$$\text{Match Percentage} = \max\left(0, \min\left(100, \left(1.0 - \frac{\text{Distance}}{2.0}\right) \times 100\right)\right)$$

---

## 📊 Evaluation & Validation Benchmarks

Evaluated across the 100 benchmark tracks with $K$-Means clustering ($K = 2 \dots 8$):

### 1. Optimal Cluster Count Evaluation
| Number of Clusters ($K$) | Inertia (SSE) | Silhouette Score | Davies-Bouldin Index | Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **$K = 2$** | 467.51 | **0.3714** | 1.182 | Broad separation (Acoustic/Ambient vs High-Energy Beats) |
| **$K = 3$** | 403.49 | 0.2401 | 1.341 | Moderate partition |
| **$K = 4$** | **355.43** | **0.3009** | **1.229** | **Recommended Default (Pop/EDM, Ballad, Hip-Hop, Rock)** |
| **$K = 5$** | 314.73 | 0.2417 | 1.305 | Granular genre division |
| **$K = 6$** | 280.44 | 0.2251 | 1.312 | Sub-genres emerging |
| **$K = 7$** | 254.77 | 0.2221 | 1.288 | Over-segmentation risk |
| **$K = 8$** | 231.25 | 0.2107 | 1.259 | Diminishing returns (Elbow plateau) |

### 2. Inference Latency (Local Host single query)
| Execution Stage | Latency |
| :--- | :---: |
| Feature Extraction & Normalization | < 0.2 ms |
| K-Means Clustering ($K=4, 100$ songs) | ~1.8 ms |
| Pairwise Distance Matrix & Sorting ($N=5$) | ~0.4 ms |
| Full End-to-End API Response Time | **~2.4 ms** |

---

## 🗂️ Repository Structure

```
261-IntelligentSystems-Application-Ch6/
├── api/
│   └── index.py               # Vercel serverless WSGI entry point
├── data/
│   └── spotify_songs.json     # Curated 100-track benchmark dataset with 8D audio features & 2D PCA
├── static/
│   ├── css/
│   │   └── style.css          # Spotify Dark & Neon responsive stylesheet
│   └── js/
│       └── script.js          # Interactive 2D Canvas, Radar Visualizer & REST client
├── templates/
│   └── index.html             # Studio user interface layout
├── prepare_data.py            # Data preprocessing, standardization, and PCA pipeline
├── app.py                     # High-performance Flask server with REST API
├── test_app.py                # Automated test suite (9 test cases, 100% pass)
├── requirements.txt           # Python library dependencies
├── vercel.json                # Vercel serverless rewrite rules
└── README.md                  # Comprehensive academic & practical documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/phenomenal291/261-IntelligentSystems-Application-Ch6.git
cd 261-IntelligentSystems-Application-Ch6
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Recompile the Spotify Dataset
```bash
python prepare_data.py
```

### 4. Launch the Web Studio
```bash
python app.py
```
Open **`http://127.0.0.1:5005`** in your web browser.

---

## 🎓 Academic Context

Developed for **CO3061: Intelligent Systems (Hệ thống Thông minh)** at **Ho Chi Minh City University of Technology (HCMUT)**.
- **Topic**: Chapter 6 - Clustering & Validation (K-Means, Agglomerative Hierarchical, DBSCAN).
- **Practical Application**: Music Recommendation System (Spotify Audio Feature Similarity).

