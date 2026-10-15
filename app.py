"""
Spotify Song Recommendation System using Unsupervised Clustering
Chapter 6 Web Application - HK261 Intelligent Systems
Algorithms: K-Means, Agglomerative Hierarchical, and DBSCAN
Features: 2D Audio Space PCA Canvas, Audio Radar Comparison, Cluster Validation & Live Recommendation Engine
"""

import os
import io
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "spotify_songs.json")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# WSGI Prefix Middleware for Vercel Serverless Routing
class PrefixMiddleware(object):
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        forwarded_uri = (
            environ.get("HTTP_X_FORWARDED_URI")
            or environ.get("HTTP_X_NOW_ROUTE_MATCHES")
            or environ.get("RAW_URI")
            or environ.get("REQUEST_URI")
        )
        if forwarded_uri and environ.get("PATH_INFO") in ["/api/index.py", "/api/index", "/api"]:
            clean_path = forwarded_uri.split("?")[0]
            if clean_path:
                environ["PATH_INFO"] = clean_path

        path = environ.get("PATH_INFO", "")
        while "//" in path:
            path = path.replace("//", "/")
        environ["PATH_INFO"] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = PrefixMiddleware(app.wsgi_app)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# Distinct Vibrant Color Palette for Audio Clusters (Dark Theme friendly)
CLUSTER_COLORS = [
    "#1DB954",  # Spotify Green
    "#38BDF8",  # Sky Blue
    "#F43F5E",  # Rose Red
    "#F59E0B",  # Amber Yellow
    "#A855F7",  # Purple
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald Green
    "#F97316",  # Vibrant Orange
    "#6366F1"   # Indigo
]
NOISE_COLOR = "#64748B"  # Slate Grey for DBSCAN Outliers / Noise

# Load Pre-compiled Spotify Dataset
print(f"Loading Spotify Songs Dataset from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    # If not compiled yet, generate it
    from prepare_data import main as generate_dataset
    generate_dataset()

with open(DATA_FILE, "r", encoding="utf-8") as f:
    DATASET_PAYLOAD = json.load(f)

TRACKS = DATASET_PAYLOAD["tracks"]
FEATURE_COLS = DATASET_PAYLOAD["feature_columns"]
X_RAW = np.array([[t[col] for col in FEATURE_COLS] for t in TRACKS], dtype=float)
X_FEATURE_VECTORS = np.array([t["feature_vector"] for t in TRACKS], dtype=float)
PCA_COORDS = np.array([[t["pca_x"], t["pca_y"]] for t in TRACKS], dtype=float)

# Fast Lookup Maps
TRACK_MAP = {t["id"]: t for t in TRACKS}
TRACK_INDEX_MAP = {t["id"]: i for i, t in enumerate(TRACKS)}
CURATED_SAMPLES = [t for t in TRACKS if t.get("curated", False)]

print(f"Spotify Dataset Loaded: {len(TRACKS)} tracks, {len(FEATURE_COLS)} features, {len(CURATED_SAMPLES)} curated samples.")


def run_clustering(algorithm="kmeans", k=4, init="k-means++", linkage="ward", eps=1.8, min_samples=3):
    """
    Executes clustering on the standardized Spotify feature vectors.
    Returns: labels, centroids_2d, centroids_8d, metrics, explanation
    """
    X = X_FEATURE_VECTORS
    n_samples = len(X)
    point_types = ["normal"] * n_samples
    inertia = None
    sil_score = None
    db_score = None
    explanation = ""

    if algorithm == "kmeans":
        k = max(2, min(int(k), 10))
        model = KMeans(n_clusters=k, init=init, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        n_clusters = k
        n_noise = 0
        centroids_8d = model.cluster_centers_

        # Calculate 2D centroids by projecting 8D cluster centers or mean of 2D coordinates
        centroids_2d = []
        for c in range(k):
            pts_2d = PCA_COORDS[labels == c]
            if len(pts_2d) > 0:
                centroids_2d.append([round(float(pts_2d[:, 0].mean()), 4), round(float(pts_2d[:, 1].mean()), 4)])
            else:
                centroids_2d.append([0.0, 0.0])

        explanation = (
            f"K-Means grouped {n_samples} songs into {k} clusters by minimizing the within-cluster Sum of Squared Errors (SSE = {inertia:.1f}). "
            f"Initialization method '{init}' ensures separated seed centers. Each song is partitioned into its closest centroid in 8D audio space."
        )

    elif algorithm == "hierarchical":
        k = max(2, min(int(k), 10))
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)
        n_clusters = k
        n_noise = 0

        centroids_2d = []
        centroids_8d = []
        for c in range(k):
            pts_2d = PCA_COORDS[labels == c]
            pts_8d = X[labels == c]
            if len(pts_2d) > 0:
                centroids_2d.append([round(float(pts_2d[:, 0].mean()), 4), round(float(pts_2d[:, 1].mean()), 4)])
                centroids_8d.append(pts_8d.mean(axis=0))
            else:
                centroids_2d.append([0.0, 0.0])
                centroids_8d.append(np.zeros(X.shape[1]))

        centroids_8d = np.array(centroids_8d)

        linkage_desc = {
            "ward": "Ward's criterion (minimizes total variance within clusters, producing compact, balanced musical genres)",
            "complete": "Complete Link / MAX (uses greatest pairwise distance, preventing elongated chains)",
            "average": "Average Link / MEAN (uses average pairwise distance across all songs)",
            "single": "Single Link / MIN (infamous for the 'chaining effect', linking disparate tracks through bridge songs)"
        }.get(linkage, linkage)

        explanation = (
            f"Agglomerative Hierarchical Clustering merged songs bottom-up into a dendrogram tree with {k} clusters using {linkage_desc}. "
            "Notice how cluster groupings reflect acoustic similarity hierarchies."
        )

    elif algorithm == "dbscan":
        eps = float(eps)
        min_samples = int(min_samples)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        core_indices = set(model.core_sample_indices_)
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int(np.sum(labels == -1))

        for i in range(n_samples):
            if labels[i] == -1:
                point_types[i] = "noise"
            elif i in core_indices:
                point_types[i] = "core"
            else:
                point_types[i] = "border"

        centroids_2d = []
        centroids_8d = []
        for c in sorted(list(unique_labels - {-1})):
            pts_2d = PCA_COORDS[labels == c]
            pts_8d = X[labels == c]
            if len(pts_2d) > 0:
                centroids_2d.append([round(float(pts_2d[:, 0].mean()), 4), round(float(pts_2d[:, 1].mean()), 4)])
                centroids_8d.append(pts_8d.mean(axis=0))

        centroids_8d = np.array(centroids_8d) if len(centroids_8d) > 0 else np.zeros((0, X.shape[1]))

        explanation = (
            f"DBSCAN detected {n_clusters} dense core musical cluster(s) and flagged {n_noise} outlier song(s) as Noise "
            f"(Eps={eps:.2f}, MinPts={min_samples}). Outliers typically correspond to distinct avant-garde, minimal ambient, or classical piano compositions!"
        )

    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    # Calculate Validation Metrics
    valid_clusters = set(labels) - {-1}
    if len(valid_clusters) >= 2 and len(valid_clusters) < n_samples:
        mask = labels != -1 if algorithm == "dbscan" else np.ones(n_samples, dtype=bool)
        if np.sum(mask) > len(valid_clusters):
            try:
                sil_score = float(silhouette_score(X[mask], labels[mask]))
                db_score = float(davies_bouldin_score(X[mask], labels[mask]))
            except Exception:
                pass

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": round(sil_score, 4) if sil_score is not None else "N/A",
        "inertia": round(inertia, 2) if inertia is not None else "N/A",
        "davies_bouldin": round(db_score, 4) if db_score is not None else "N/A"
    }

    return labels, point_types, centroids_2d, centroids_8d, metrics, explanation


def get_recommendations(seed_id, labels, metric="cosine", top_n=5, filter_cluster=False, custom_features=None):
    """
    Computes top-N song recommendations based on distance in standardized audio feature space.
    """
    X = X_FEATURE_VECTORS

    if custom_features is not None:
        # Custom user-tuned audio feature vector
        scaler_mean = np.array(DATASET_PAYLOAD["scaler_mean"])
        scaler_scale = np.array(DATASET_PAYLOAD["scaler_scale"])
        seed_raw = np.array(custom_features, dtype=float)
        seed_vec = (seed_raw - scaler_mean) / scaler_scale
        seed_song = {
            "id": "custom",
            "title": "Custom Tuned Track",
            "artist": "User Profile",
            "genre": "User Preference",
            "year": 2026,
            "danceability": float(seed_raw[0]),
            "energy": float(seed_raw[1]),
            "valence": float(seed_raw[2]),
            "tempo": float(seed_raw[3]),
            "acousticness": float(seed_raw[4]),
            "instrumentalness": float(seed_raw[5]),
            "speechiness": float(seed_raw[6]),
            "loudness": float(seed_raw[7]),
            "popularity": 99,
            "curated": False,
            "feature_vector": seed_vec.tolist()
        }
        seed_idx = -1
        seed_cluster = -1
    else:
        seed_idx = TRACK_INDEX_MAP.get(seed_id, 0)
        seed_song = TRACKS[seed_idx]
        seed_vec = X[seed_idx]
        seed_cluster = int(labels[seed_idx])

    # Distance computation
    dists = pairwise_distances(seed_vec.reshape(1, -1), X, metric=metric)[0]

    # Candidate indices
    candidate_indices = []
    for i in range(len(TRACKS)):
        if seed_idx >= 0 and i == seed_idx:
            continue  # Do not recommend the seed song itself
        if filter_cluster and seed_cluster != -1 and labels[i] != seed_cluster:
            continue  # Restrict to same cluster
        candidate_indices.append(i)

    if not candidate_indices:
        candidate_indices = [i for i in range(len(TRACKS)) if i != seed_idx]

    # Sort candidates by distance
    sorted_candidates = sorted(candidate_indices, key=lambda idx: dists[idx])[:top_n]

    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0

    recommendations = []
    for rank, idx in enumerate(sorted_candidates, start=1):
        target = TRACKS[idx]
        d = float(dists[idx])
        c_id = int(labels[idx])

        # Match percentage calculation based on distance metric
        if metric == "cosine":
            match_pct = max(0.0, min(100.0, round((1.0 - d / 2.0) * 100.0, 1)))
        else:
            match_pct = max(0.0, min(100.0, round((1.0 - (d / (max_dist + 1e-5))) * 100.0, 1)))

        # Reason for recommendation
        same_cluster_note = "in the same audio mood cluster" if c_id == seed_cluster and c_id != -1 else "across closely aligned acoustic dimensions"
        reason = f"High {match_pct}% audio similarity {same_cluster_note} with complementary Tempo ({target['tempo']:.0f} BPM) & Energy ({target['energy']:.2f})."

        recommendations.append({
            "rank": rank,
            "id": target["id"],
            "title": target["title"],
            "artist": target["artist"],
            "genre": target["genre"],
            "year": target["year"],
            "tempo": target["tempo"],
            "danceability": target["danceability"],
            "energy": target["energy"],
            "valence": target["valence"],
            "acousticness": target["acousticness"],
            "instrumentalness": target["instrumentalness"],
            "speechiness": target["speechiness"],
            "cluster": c_id,
            "distance": round(d, 4),
            "match_percentage": match_pct,
            "pca_x": target["pca_x"],
            "pca_y": target["pca_y"],
            "reason": reason
        })

    # Radar Comparison Payload (6 key normalized radar dimensions)
    # Dimensions: [Danceability, Energy, Valence, Acousticness, Instrumentalness, Speechiness]
    radar_labels = ["Danceability", "Energy", "Valence", "Acousticness", "Instrumentalness", "Speechiness"]
    seed_radar = [
        round(seed_song["danceability"], 3),
        round(seed_song["energy"], 3),
        round(seed_song["valence"], 3),
        round(seed_song["acousticness"], 3),
        round(seed_song["instrumentalness"], 3),
        round(seed_song["speechiness"], 3)
    ]

    top_rec = recommendations[0] if recommendations else seed_song
    rec_radar = [
        round(top_rec["danceability"], 3),
        round(top_rec["energy"], 3),
        round(top_rec["valence"], 3),
        round(top_rec["acousticness"], 3),
        round(top_rec["instrumentalness"], 3),
        round(top_rec["speechiness"], 3)
    ]

    radar_payload = {
        "labels": radar_labels,
        "seed_title": seed_song["title"],
        "seed_artist": seed_song["artist"],
        "seed_values": seed_radar,
        "rec_title": top_rec["title"],
        "rec_artist": top_rec["artist"],
        "rec_values": rec_radar
    }

    return recommendations, seed_song, radar_payload


# Pre-compute initial state (K-Means K=4, seed: Blinding Lights)
DEFAULT_LABELS, DEFAULT_TYPES, DEFAULT_C2D, DEFAULT_C8D, DEFAULT_METRICS, DEFAULT_EXPL = run_clustering(
    algorithm="kmeans", k=4, init="k-means++"
)
DEFAULT_RECS, DEFAULT_SEED, DEFAULT_RADAR = get_recommendations(
    seed_id="sp_01", labels=DEFAULT_LABELS, metric="cosine", top_n=5
)

# Pre-packaged initial points payload
INITIAL_POINTS = []
for i, t in enumerate(TRACKS):
    c_id = int(DEFAULT_LABELS[i])
    color = NOISE_COLOR if c_id == -1 else CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
    INITIAL_POINTS.append({
        "id": t["id"],
        "title": t["title"],
        "artist": t["artist"],
        "genre": t["genre"],
        "year": t["year"],
        "tempo": t["tempo"],
        "danceability": t["danceability"],
        "energy": t["energy"],
        "x": t["pca_x"],
        "y": t["pca_y"],
        "cluster": c_id,
        "type": DEFAULT_TYPES[i],
        "color": color
    })


@app.route("/", methods=["GET", "POST", "OPTIONS"])
@app.route("/api", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/index", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/index.py", methods=["GET", "POST", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    return render_template(
        "index.html",
        tracks=TRACKS,
        curated_samples=CURATED_SAMPLES,
        initial_points=INITIAL_POINTS,
        initial_centroids=DEFAULT_C2D,
        initial_metrics=DEFAULT_METRICS,
        initial_seed=DEFAULT_SEED,
        initial_recommendations=DEFAULT_RECS,
        initial_radar=DEFAULT_RADAR,
        initial_explanation=DEFAULT_EXPL,
        cluster_colors=CLUSTER_COLORS
    )


@app.route("/api/songs", methods=["GET", "OPTIONS"])
def get_songs():
    """Returns catalog of songs and curated samples."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    return jsonify({
        "success": True,
        "tracks": TRACKS,
        "curated_samples": CURATED_SAMPLES,
        "total": len(TRACKS)
    })


@app.route("/api/cluster", methods=["POST", "OPTIONS"])
def api_cluster():
    """Performs dynamic clustering on the Spotify dataset."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        t0 = time.perf_counter()
        data = request.get_json(silent=True) or {}

        algorithm = data.get("algorithm", "kmeans")
        k = int(data.get("k", 4))
        init_method = data.get("init", "k-means++")
        linkage = data.get("linkage", "ward")
        eps = float(data.get("eps", 1.8))
        min_samples = int(data.get("min_samples", 3))

        seed_id = data.get("seed_id", "sp_01")
        metric = data.get("metric", "cosine")
        top_n = int(data.get("top_n", 5))
        filter_cluster = bool(data.get("filter_cluster", False))
        custom_features = data.get("custom_features", None)

        # 1. Run Clustering
        labels, point_types, centroids_2d, centroids_8d, metrics, explanation = run_clustering(
            algorithm=algorithm, k=k, init=init_method, linkage=linkage, eps=eps, min_samples=min_samples
        )

        # 2. Compute Recommendations
        recommendations, seed_song, radar_payload = get_recommendations(
            seed_id=seed_id,
            labels=labels,
            metric=metric,
            top_n=top_n,
            filter_cluster=filter_cluster,
            custom_features=custom_features
        )

        # 3. Format points with updated cluster colors
        points_payload = []
        for i, t in enumerate(TRACKS):
            c_id = int(labels[i])
            color = NOISE_COLOR if c_id == -1 else CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
            points_payload.append({
                "id": t["id"],
                "title": t["title"],
                "artist": t["artist"],
                "genre": t["genre"],
                "year": t["year"],
                "tempo": t["tempo"],
                "danceability": t["danceability"],
                "energy": t["energy"],
                "x": t["pca_x"],
                "y": t["pca_y"],
                "cluster": c_id,
                "type": point_types[i],
                "color": color
            })

        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics["inference_time_ms"] = inference_time_ms

        return jsonify({
            "success": True,
            "algorithm": algorithm,
            "points": points_payload,
            "centroids": centroids_2d,
            "metrics": metrics,
            "seed_song": seed_song,
            "recommendations": recommendations,
            "radar": radar_payload,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def api_recommend():
    """Generates recommendations for a chosen seed song or custom vector."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        t0 = time.perf_counter()
        data = request.get_json(silent=True) or {}

        seed_id = data.get("seed_id", "sp_01")
        algorithm = data.get("algorithm", "kmeans")
        k = int(data.get("k", 4))
        linkage = data.get("linkage", "ward")
        eps = float(data.get("eps", 1.8))
        min_samples = int(data.get("min_samples", 3))
        metric = data.get("metric", "cosine")
        top_n = int(data.get("top_n", 5))
        filter_cluster = bool(data.get("filter_cluster", False))
        custom_features = data.get("custom_features", None)

        labels, point_types, centroids_2d, _, metrics, explanation = run_clustering(
            algorithm=algorithm, k=k, linkage=linkage, eps=eps, min_samples=min_samples
        )

        recommendations, seed_song, radar_payload = get_recommendations(
            seed_id=seed_id,
            labels=labels,
            metric=metric,
            top_n=top_n,
            filter_cluster=filter_cluster,
            custom_features=custom_features
        )

        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return jsonify({
            "success": True,
            "seed_song": seed_song,
            "recommendations": recommendations,
            "radar": radar_payload,
            "metrics": metrics,
            "inference_time_ms": inference_time_ms
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/validation-curve", methods=["POST", "OPTIONS"])
def api_validation_curve():
    """Computes Elbow (SSE vs K) and Silhouette curves for K in [2, 8]."""
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        X = X_FEATURE_VECTORS
        k_values = list(range(2, 9))
        elbow_curve = []
        silhouette_curve = []

        for k in k_values:
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10).fit(X)
            elbow_curve.append(round(float(km.inertia_), 2))
            score = silhouette_score(X, km.labels_)
            silhouette_curve.append(round(float(score), 4))

        best_k = int(k_values[np.argmax(silhouette_curve)])

        return jsonify({
            "success": True,
            "k_values": k_values,
            "elbow_curve": elbow_curve,
            "silhouette_curve": silhouette_curve,
            "best_k": best_k
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/static/<path:filename>")
@app.route("/api/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print(f"🎵 Starting Spotify Song Recommendation Studio at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
