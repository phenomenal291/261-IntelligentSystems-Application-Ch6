"""
Automated Test Suite for Spotify Song Recommendation Studio
HK261 Intelligent Systems - Chapter 6 Application
Tests all clustering algorithms, distance metrics, seed selections, and validation curves.
"""

import unittest
import json
import numpy as np
from app import app

class TestSpotifyRecommendationStudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_index_page(self):
        """Verify homepage loads with HTTP 200 and required elements."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn('Spotify Song Recommender', res.text)
        self.assertIn('playlist-table', res.text)
        self.assertIn('audioSpaceCanvas', res.text)

    def test_02_seed_recommendations_logic(self):
        """Test song recommendations across distinct musical genres."""
        test_cases = [
            ("sp_01", "Blinding Lights", "Pop/Fast BPM", ["Yellow", "Stay", "Wonderwall", "Mr. Brightside"]),
            ("sp_24", "Someone Like You", "Acoustic Ballad", ["Easy On Me", "Drivers License", "Lovely", "All of Me"]),
            ("sp_45", "Smells Like Teen Spirit", "Grunge Rock", ["Sweet Child O' Mine", "In the End", "Believer"]),
            ("sp_34", "HUMBLE.", "Hip-Hop", ["Without Me", "Not Like Us", "Lose Yourself", "SICKO MODE"])
        ]

        print("\n--- Testing Recommendation Quality across Genres ---")
        for seed_id, title, genre_desc, expected_similar in test_cases:
            res = self.client.post('/api/cluster', json={
                'algorithm': 'kmeans', 'k': 4, 'seed_id': seed_id, 'metric': 'cosine', 'top_n': 5
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            recs = data['recommendations']
            self.assertEqual(len(recs), 5)

            rec_titles = [r['title'] for r in recs]
            print(f"\nSeed: {title} ({genre_desc})")
            for i, r in enumerate(recs[:3], 1):
                print(f"  #{i}: {r['title']} by {r['artist']} ({r['match_percentage']}%) - {r['genre']} [{r['tempo']:.0f} BPM, Energy: {r['energy']}]")

            # Check that at least one of expected similar songs is present in top 5
            matches = any(exp in rec_titles for exp in expected_similar)
            self.assertTrue(matches, f"None of {expected_similar} found in {rec_titles} for {title}")

    def test_03_kmeans_monotonic_sse(self):
        """Verify SSE decreases as K increases (fundamental K-Means property)."""
        print("\n--- Testing K-Means SSE and Silhouette across K in [2, 8] ---")
        sses = []
        for k in [2, 3, 4, 5, 8]:
            res = self.client.post('/api/cluster', json={'algorithm': 'kmeans', 'k': k, 'init': 'k-means++'})
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            sse = d['metrics']['inertia']
            sil = d['metrics']['silhouette']
            sses.append(sse)
            print(f"  K={k}: SSE = {sse:.1f}, Silhouette = {sil}")
            self.assertEqual(d['metrics']['n_clusters'], k)
            self.assertTrue(0.1 <= sil <= 1.0)

        # Monotonically decreasing
        self.assertEqual(sses, sorted(sses, reverse=True), "Inertia (SSE) must decrease monotonically with K!")

    def test_04_hierarchical_linkages(self):
        """Verify Hierarchical clustering with Ward, Complete, Average, and Single link."""
        print("\n--- Testing Hierarchical Linkages ---")
        for link in ['ward', 'complete', 'average', 'single']:
            res = self.client.post('/api/cluster', json={'algorithm': 'hierarchical', 'k': 4, 'linkage': link})
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            clusters = set(p['cluster'] for p in d['points'])
            sil = d['metrics']['silhouette']
            print(f"  Linkage '{link:8s}': {len(clusters)} clusters, Silhouette = {sil}")
            self.assertEqual(len(clusters), 4)

    def test_05_dbscan_clustering_and_outliers(self):
        """Verify DBSCAN identifies clusters and isolates ambient/classical outliers."""
        print("\n--- Testing DBSCAN & Outlier Detection ---")
        res = self.client.post('/api/cluster', json={'algorithm': 'dbscan', 'eps': 1.8, 'min_samples': 3})
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        n_clusters = d['metrics']['n_clusters']
        n_noise = d['metrics']['n_noise']
        print(f"  DBSCAN (eps=1.8, MinPts=3): {n_clusters} clusters, {n_noise} noise points")
        self.assertTrue(n_clusters >= 1)
        self.assertTrue(n_noise > 0)

        # Verify classical piano / ambient tracks are classified as noise (-1)
        noise_tracks = [p['title'] for p in d['points'] if p['type'] == 'noise']
        print(f"  Sample Outliers: {noise_tracks[:5]}")
        has_ambient = any(name in noise_tracks for name in ["Gymnopédie No. 1", "Weightless", "Clair de Lune", "Nuvole Bianche"])
        self.assertTrue(has_ambient, "Ambient/classical tracks should logically be identified as noise outliers in mainstream dance/pop space!")

    def test_06_distance_metrics(self):
        """Verify recommendation outputs for Cosine, Euclidean, and Manhattan metrics."""
        print("\n--- Testing Distance Metrics ---")
        for metric in ['cosine', 'euclidean', 'manhattan']:
            res = self.client.post('/api/recommend', json={'seed_id': 'sp_01', 'metric': metric, 'top_n': 3})
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            recs = d['recommendations']
            self.assertEqual(len(recs), 3)
            print(f"  Metric '{metric:9s}': Top #1 is '{recs[0]['title']}' with distance {recs[0]['distance']} ({recs[0]['match_percentage']}%)")

    def test_07_cluster_filter_toggle(self):
        """Verify that intra-cluster filtering strictly yields songs from the same cluster."""
        res_unfiltered = self.client.post('/api/recommend', json={'seed_id': 'sp_01', 'filter_cluster': False, 'top_n': 5})
        d_unf = res_unfiltered.get_json()

        res_filtered = self.client.post('/api/recommend', json={'seed_id': 'sp_01', 'filter_cluster': True, 'top_n': 5})
        d_fil = res_filtered.get_json()

        self.assertEqual(len(d_fil['recommendations']), 5)
        # All recommendations in filtered mode must share the seed's cluster
        seed_cluster = d_fil['recommendations'][0]['cluster']
        for r in d_fil['recommendations']:
            self.assertEqual(r['cluster'], seed_cluster)
        print(f"\n--- Cluster Filter Verified: All top recommendations belong strictly to Cluster #{seed_cluster + 1} ---")

    def test_08_custom_audio_features(self):
        """Verify recommendation with user-tuned custom audio profile."""
        # Custom high-energy EDM profile
        custom_vec = [0.85, 0.95, 0.80, 130.0, 0.01, 0.0, 0.05, -3.5]
        res = self.client.post('/api/recommend', json={'custom_features': custom_vec, 'metric': 'cosine', 'top_n': 3})
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        print("\n--- Custom Tuned Profile (130 BPM, Energy: 0.95, Dance: 0.85) ---")
        for i, r in enumerate(d['recommendations'], 1):
            print(f"  #{i}: {r['title']} - {r['genre']} [{r['tempo']:.0f} BPM, Energy: {r['energy']:.2f}]")
        self.assertEqual(d['seed_song']['id'], 'custom')

    def test_09_validation_curves(self):
        """Verify validation curve computation for Elbow and Silhouette."""
        print("\n--- Testing Validation Curves Endpoint ---")
        res = self.client.post('/api/validation-curve', json={})
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d['k_values'], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(d['elbow_curve']), 7)
        self.assertEqual(len(d['silhouette_curve']), 7)
        self.assertIn(d['best_k'], d['k_values'])
        print(f"  Evaluated K: {d['k_values']}")
        print(f"  Elbow curve: {d['elbow_curve']}")
        print(f"  Silhouette curve: {d['silhouette_curve']}")
        print(f"  Optimal K recommended: {d['best_k']} (Silhouette = {max(d['silhouette_curve'])})")

if __name__ == '__main__':
    unittest.main()

