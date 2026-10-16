/**
 * Spotify Song Recommendation Studio - Client Controller & ML Engine
 * HK261 Intelligent Systems - Chapter 6 Application
 * Algorithms: K-Means, Agglomerative Hierarchical, DBSCAN
 * Visualizations: 2D Audio Feature Space (PCA), Radar Chart, Cluster Validation Curves
 *
 * Fully dual-mode: Real-time 0ms client-side execution + server API sync.
 */

(function () {
    "use strict";

    // Application State
    const state = {
        algorithm: "kmeans",
        k: 4,
        init: "k-means++",
        linkage: "ward",
        eps: 1.8,
        minSamples: 3,
        seedId: "sp_01",
        metric: "cosine",
        topN: 5,
        filterCluster: false,
        customFeatures: null,
        tracks: [],
        points: [],
        centroids: [],
        metrics: {},
        seedSong: null,
        recommendations: [],
        radar: null,
        clusterColors: [
            "#1DB954", "#38BDF8", "#F43F5E", "#F59E0B",
            "#A855F7", "#EC4899", "#06B6D4", "#10B981",
            "#F97316", "#6366F1"
        ],
        hoveredPoint: null,
        animFrameId: null,
        pulseAngle: 0,
        validationCurves: null
    };

    // DOM Elements
    const elements = {
        // Step 1: Seed Selection
        songSelect: document.getElementById("songSelect"),
        randomSongBtn: document.getElementById("randomSongBtn"),
        bannerTitle: document.getElementById("bannerTitle"),
        bannerArtist: document.getElementById("bannerArtist"),
        bannerGenre: document.getElementById("bannerGenre"),
        bannerTempo: document.getElementById("bannerTempo"),

        // Custom Tuning
        customTuningToggle: document.getElementById("customTuningToggle"),
        sliderDance: document.getElementById("sliderDance"),
        sliderDanceVal: document.getElementById("sliderDanceVal"),
        sliderEnergy: document.getElementById("sliderEnergy"),
        sliderEnergyVal: document.getElementById("sliderEnergyVal"),
        sliderValence: document.getElementById("sliderValence"),
        sliderValenceVal: document.getElementById("sliderValenceVal"),
        sliderTempo: document.getElementById("sliderTempo"),
        sliderTempoVal: document.getElementById("sliderTempoVal"),
        sliderAcoustic: document.getElementById("sliderAcoustic"),
        sliderAcousticVal: document.getElementById("sliderAcousticVal"),
        applyCustomBtn: document.getElementById("applyCustomBtn"),

        // Step 2: Algorithm Controls
        algoTabs: document.querySelectorAll(".tab-btn"),
        kmeansControls: document.getElementById("kmeansControls"),
        hierarchicalControls: document.getElementById("hierarchicalControls"),
        dbscanControls: document.getElementById("dbscanControls"),

        kSlider: document.getElementById("kSlider"),
        kVal: document.getElementById("kVal"),
        initBtns: document.querySelectorAll("[data-init]"),

        hierKSlider: document.getElementById("hierKSlider"),
        hierKVal: document.getElementById("hierKVal"),
        linkageSelect: document.getElementById("linkageSelect"),

        epsSlider: document.getElementById("epsSlider"),
        epsVal: document.getElementById("epsVal"),
        minPtsSlider: document.getElementById("minPtsSlider"),
        minPtsVal: document.getElementById("minPtsVal"),

        // Step 3: Recommendation Settings
        metricBtns: document.querySelectorAll("[data-metric]"),
        filterClusterCheckbox: document.getElementById("filterClusterCheckbox"),
        topNSlider: document.getElementById("topNSlider"),
        topNVal: document.getElementById("topNVal"),

        // Action: Validation Curves Button
        openCurveBtn: document.getElementById("openCurveBtn"),

        // KPI Badges
        metricClusters: document.getElementById("metricClusters"),
        metricSilhouette: document.getElementById("metricSilhouette"),
        metricSilBadge: document.getElementById("metricSilBadge"),
        metricInertia: document.getElementById("metricInertia"),
        metricNoise: document.getElementById("metricNoise"),

        // 2D Canvas & Tooltip
        audioSpaceCanvas: document.getElementById("audioSpaceCanvas"),
        canvasTooltip: document.getElementById("canvasTooltip"),
        canvasLegend: document.getElementById("canvasLegend"),

        // Seed Profile & Radar
        profileSeedTitle: document.getElementById("profileSeedTitle"),
        profileSeedArtist: document.getElementById("profileSeedArtist"),
        barSeedDance: document.getElementById("barSeedDance"),
        valSeedDance: document.getElementById("valSeedDance"),
        barSeedEnergy: document.getElementById("barSeedEnergy"),
        valSeedEnergy: document.getElementById("valSeedEnergy"),
        barSeedValence: document.getElementById("barSeedValence"),
        valSeedValence: document.getElementById("valSeedValence"),
        barSeedAcoustic: document.getElementById("barSeedAcoustic"),
        valSeedAcoustic: document.getElementById("valSeedAcoustic"),
        radarCanvas: document.getElementById("radarCanvas"),
        radarSeedLegend: document.getElementById("radarSeedLegend"),
        radarRecLegend: document.getElementById("radarRecLegend"),

        // Recommendations List
        recsGrid: document.getElementById("recsGrid"),
        recCountBadge: document.getElementById("recCountBadge"),
        recLatency: document.getElementById("recLatency"),

        // Pedagogical Insight
        explanationText: document.getElementById("explanationText"),

        // Validation Modal
        curveModal: document.getElementById("curveModal"),
        closeCurveBtn: document.getElementById("closeCurveBtn"),
        elbowChartCanvas: document.getElementById("elbowChart"),
        silhouetteChartCanvas: document.getElementById("silhouetteChart"),
        modalBestKText: document.getElementById("modalBestKText")
    };

    let elbowChartInstance = null;
    let silChartInstance = null;
    let debounceTimer = null;

    // --- High-Performance Client Machine Learning Module ---

    function euclideanDist(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    function cosineDist(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        if (denom === 0) return 1.0;
        const sim = Math.max(-1.0, Math.min(1.0, dot / denom));
        return 1.0 - sim;
    }

    function manhattanDist(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    function calcDistance(a, b, metric) {
        if (metric === "cosine") return cosineDist(a, b);
        if (metric === "manhattan") return manhattanDist(a, b);
        return euclideanDist(a, b);
    }

    // Client K-Means Implementation
    function runKMeansClient(tracks, k, initMethod) {
        const N = tracks.length;
        const D = tracks[0].feature_vector.length;
        let centroids = [];

        if (initMethod === "random") {
            const indices = [];
            while (indices.length < k) {
                const idx = Math.floor(Math.random() * N);
                if (!indices.includes(idx)) indices.push(idx);
            }
            centroids = indices.map(idx => [...tracks[idx].feature_vector]);
        } else {
            // Deterministic k-means++ for consistency
            centroids.push([...tracks[0].feature_vector]);
            while (centroids.length < k) {
                const distsSq = [];
                let sumSq = 0;
                for (let i = 0; i < N; i++) {
                    let minD = Infinity;
                    for (let c = 0; c < centroids.length; c++) {
                        const d = euclideanDist(tracks[i].feature_vector, centroids[c]);
                        if (d < minD) minD = d;
                    }
                    const dSq = minD * minD;
                    distsSq.push(dSq);
                    sumSq += dSq;
                }
                let target = ((centroids.length * 37) % 100 / 100.0) * sumSq;
                let chosen = 0;
                for (let i = 0; i < N; i++) {
                    target -= distsSq[i];
                    if (target <= 0) {
                        chosen = i;
                        break;
                    }
                }
                centroids.push([...tracks[chosen].feature_vector]);
            }
        }

        let labels = new Array(N).fill(0);
        const maxIter = 25;
        for (let iter = 0; iter < maxIter; iter++) {
            let changed = false;
            for (let i = 0; i < N; i++) {
                let minD = Infinity, bestC = 0;
                for (let c = 0; c < k; c++) {
                    const d = euclideanDist(tracks[i].feature_vector, centroids[c]);
                    if (d < minD) {
                        minD = d;
                        bestC = c;
                    }
                }
                if (labels[i] !== bestC) {
                    labels[i] = bestC;
                    changed = true;
                }
            }
            if (!changed && iter > 0) break;

            const counts = new Array(k).fill(0);
            const newCentroids = Array.from({ length: k }, () => new Array(D).fill(0));
            for (let i = 0; i < N; i++) {
                const c = labels[i];
                counts[c]++;
                for (let d = 0; d < D; d++) {
                    newCentroids[c][d] += tracks[i].feature_vector[d];
                }
            }
            for (let c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (let d = 0; d < D; d++) {
                        centroids[c][d] = newCentroids[c][d] / counts[c];
                    }
                }
            }
        }

        let sse = 0;
        for (let i = 0; i < N; i++) {
            const c = labels[i];
            for (let d = 0; d < D; d++) {
                const diff = tracks[i].feature_vector[d] - centroids[c][d];
                sse += diff * diff;
            }
        }

        const centroids2D = [];
        for (let c = 0; c < k; c++) {
            let sumX = 0, sumY = 0, count = 0;
            for (let i = 0; i < N; i++) {
                if (labels[i] === c) {
                    sumX += tracks[i].pca_x;
                    sumY += tracks[i].pca_y;
                    count++;
                }
            }
            if (count > 0) {
                centroids2D.push([Number((sumX / count).toFixed(4)), Number((sumY / count).toFixed(4))]);
            } else {
                centroids2D.push([0, 0]);
            }
        }

        return { labels, centroids2D, sse: Number(sse.toFixed(2)), nClusters: k, nNoise: 0 };
    }

    // Client Hierarchical Clustering
    function runHierarchicalClient(tracks, k, linkage) {
        const N = tracks.length;
        const distMatrix = [];
        for (let i = 0; i < N; i++) {
            distMatrix[i] = [];
            for (let j = 0; j < N; j++) {
                distMatrix[i][j] = i === j ? 0 : euclideanDist(tracks[i].feature_vector, tracks[j].feature_vector);
            }
        }

        let clusters = tracks.map((_, i) => [i]);

        function clusterDist(c1, c2) {
            if (linkage === "single") {
                let minD = Infinity;
                for (let i of c1) {
                    for (let j of c2) {
                        if (distMatrix[i][j] < minD) minD = distMatrix[i][j];
                    }
                }
                return minD;
            } else if (linkage === "complete") {
                let maxD = -Infinity;
                for (let i of c1) {
                    for (let j of c2) {
                        if (distMatrix[i][j] > maxD) maxD = distMatrix[i][j];
                    }
                }
                return maxD;
            } else {
                let sumD = 0;
                for (let i of c1) {
                    for (let j of c2) {
                        sumD += distMatrix[i][j];
                    }
                }
                return sumD / (c1.length * c2.length);
            }
        }

        while (clusters.length > k) {
            let bestDist = Infinity, bestI = 0, bestJ = 1;
            for (let i = 0; i < clusters.length; i++) {
                for (let j = i + 1; j < clusters.length; j++) {
                    const d = clusterDist(clusters[i], clusters[j]);
                    if (d < bestDist) {
                        bestDist = d;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
            const merged = clusters[bestI].concat(clusters[bestJ]);
            clusters.splice(bestJ, 1);
            clusters[bestI] = merged;
        }

        const labels = new Array(N).fill(0);
        const centroids2D = [];
        for (let c = 0; c < clusters.length; c++) {
            let sumX = 0, sumY = 0;
            for (let idx of clusters[c]) {
                labels[idx] = c;
                sumX += tracks[idx].pca_x;
                sumY += tracks[idx].pca_y;
            }
            centroids2D.push([Number((sumX / clusters[c].length).toFixed(4)), Number((sumY / clusters[c].length).toFixed(4))]);
        }

        return { labels, centroids2D, sse: "N/A", nClusters: k, nNoise: 0 };
    }

    // Client DBSCAN Implementation
    function runDBSCANClient(tracks, eps, minSamples) {
        const N = tracks.length;
        const labels = new Array(N).fill(-1);
        const visited = new Array(N).fill(false);
        let clusterIdx = 0;

        function getNeighbors(i) {
            const neighbors = [];
            for (let j = 0; j < N; j++) {
                if (euclideanDist(tracks[i].feature_vector, tracks[j].feature_vector) <= eps) {
                    neighbors.push(j);
                }
            }
            return neighbors;
        }

        for (let i = 0; i < N; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            const neighbors = getNeighbors(i);
            if (neighbors.length >= minSamples) {
                labels[i] = clusterIdx;
                const queue = [...neighbors];
                for (let q = 0; q < queue.length; q++) {
                    const pt = queue[q];
                    if (!visited[pt]) {
                        visited[pt] = true;
                        const ptNeighbors = getNeighbors(pt);
                        if (ptNeighbors.length >= minSamples) {
                            for (let n of ptNeighbors) {
                                if (!queue.includes(n)) queue.push(n);
                            }
                        }
                    }
                    if (labels[pt] === -1) {
                        labels[pt] = clusterIdx;
                    }
                }
                clusterIdx++;
            }
        }

        const centroids2D = [];
        for (let c = 0; c < clusterIdx; c++) {
            let sumX = 0, sumY = 0, count = 0;
            for (let i = 0; i < N; i++) {
                if (labels[i] === c) {
                    sumX += tracks[i].pca_x;
                    sumY += tracks[i].pca_y;
                    count++;
                }
            }
            if (count > 0) {
                centroids2D.push([Number((sumX / count).toFixed(4)), Number((sumY / count).toFixed(4))]);
            }
        }

        const nNoise = labels.filter(l => l === -1).length;
        return { labels, centroids2D, sse: "N/A", nClusters: clusterIdx, nNoise };
    }

    // Client Silhouette Score Evaluation
    function calcSilhouetteClient(tracks, labels) {
        const N = tracks.length;
        const validIndices = [];
        const clusterMap = {};

        for (let i = 0; i < N; i++) {
            const c = labels[i];
            if (c !== -1) {
                validIndices.push(i);
                if (!clusterMap[c]) clusterMap[c] = [];
                clusterMap[c].push(i);
            }
        }

        const uniqueClusters = Object.keys(clusterMap);
        if (uniqueClusters.length < 2 || validIndices.length <= uniqueClusters.length) {
            return "N/A";
        }

        let totalSil = 0;
        for (let i of validIndices) {
            const myCluster = labels[i];
            const sameClusterPts = clusterMap[myCluster];
            let a = 0;
            if (sameClusterPts.length > 1) {
                let sumA = 0;
                for (let j of sameClusterPts) {
                    if (i !== j) sumA += euclideanDist(tracks[i].feature_vector, tracks[j].feature_vector);
                }
                a = sumA / (sameClusterPts.length - 1);
            }

            let b = Infinity;
            for (let c of uniqueClusters) {
                const cId = parseInt(c);
                if (cId === myCluster) continue;
                const otherPts = clusterMap[cId];
                let sumB = 0;
                for (let j of otherPts) {
                    sumB += euclideanDist(tracks[i].feature_vector, tracks[j].feature_vector);
                }
                const meanB = sumB / otherPts.length;
                if (meanB < b) b = meanB;
            }

            const denom = Math.max(a, b);
            const s = denom === 0 ? 0 : (b - a) / denom;
            totalSil += s;
        }

        return Number((totalSil / validIndices.length).toFixed(4));
    }

    // Client Recommendation Generation
    function generateRecommendationsClient(seedSong, tracks, labels, metric, topN, filterCluster) {
        const seedVec = seedSong.feature_vector;
        const seedCluster = seedSong.id === "custom" ? -1 : labels[tracks.findIndex(t => t.id === seedSong.id)];

        const scored = [];
        let maxDist = 0.0001;

        for (let i = 0; i < tracks.length; i++) {
            const t = tracks[i];
            if (t.id === seedSong.id) continue;
            const cId = labels[i];
            if (filterCluster && seedCluster !== -1 && cId !== seedCluster) continue;

            const d = calcDistance(seedVec, t.feature_vector, metric);
            if (d > maxDist) maxDist = d;
            scored.push({ track: t, dist: d, cluster: cId });
        }

        if (scored.length === 0) {
            for (let i = 0; i < tracks.length; i++) {
                const t = tracks[i];
                if (t.id === seedSong.id) continue;
                const d = calcDistance(seedVec, t.feature_vector, metric);
                scored.push({ track: t, dist: d, cluster: labels[i] });
            }
        }

        scored.sort((a, b) => a.dist - b.dist);
        const topCandidates = scored.slice(0, topN);

        const recs = topCandidates.map((item, idx) => {
            const t = item.track;
            let matchPct = 0;
            if (metric === "cosine") {
                matchPct = Math.max(0, Math.min(100, Math.round((1.0 - item.dist / 2.0) * 1000) / 10));
            } else {
                matchPct = Math.max(0, Math.min(100, Math.round((1.0 - item.dist / (maxDist + 1e-5)) * 1000) / 10));
            }

            const reason = (item.cluster === seedCluster && seedCluster !== -1)
                ? `Same cluster #${item.cluster + 1} • ${t.genre} • ${Math.round(t.tempo)} BPM`
                : `Acoustic similarity • ${t.genre} • ${Math.round(t.tempo)} BPM`;

            return {
                rank: idx + 1,
                id: t.id,
                title: t.title,
                artist: t.artist,
                genre: t.genre,
                year: t.year,
                tempo: t.tempo,
                danceability: t.danceability,
                energy: t.energy,
                valence: t.valence,
                acousticness: t.acousticness,
                instrumentalness: t.instrumentalness,
                speechiness: t.speechiness,
                cluster: item.cluster,
                distance: Number(item.dist.toFixed(4)),
                match_percentage: matchPct,
                pca_x: t.pca_x,
                pca_y: t.pca_y,
                reason: reason
            };
        });

        const topRec = recs[0] || seedSong;
        const radar = {
            labels: ["Danceability", "Energy", "Valence", "Acousticness", "Instrumentalness", "Speechiness"],
            seed_title: seedSong.title,
            seed_artist: seedSong.artist,
            seed_values: [
                seedSong.danceability, seedSong.energy, seedSong.valence,
                seedSong.acousticness, seedSong.instrumentalness, seedSong.speechiness
            ],
            rec_title: topRec.title,
            rec_artist: topRec.artist,
            rec_values: [
                topRec.danceability, topRec.energy, topRec.valence,
                topRec.acousticness, topRec.instrumentalness, topRec.speechiness
            ]
        };

        return { recommendations: recs, radar };
    }

    // Initialize with Pre-computed Server Data
    function init() {
        if (window.INITIAL_DATA) {
            state.tracks = window.INITIAL_DATA.tracks || [];
            state.points = window.INITIAL_DATA.points || [];
            state.centroids = window.INITIAL_DATA.centroids || [];
            state.metrics = window.INITIAL_DATA.metrics || {};
            state.seedSong = window.INITIAL_DATA.seed_song || null;
            state.recommendations = window.INITIAL_DATA.recommendations || [];
            state.radar = window.INITIAL_DATA.radar || null;
            if (window.INITIAL_DATA.cluster_colors && window.INITIAL_DATA.cluster_colors.length > 0) {
                state.clusterColors = window.INITIAL_DATA.cluster_colors;
            }

            if (state.seedSong) {
                state.seedId = state.seedSong.id;
            }
        }

        // If tracks was not explicitly loaded, build it from points
        if (state.tracks.length === 0 && state.points.length > 0) {
            state.tracks = state.points.map(p => ({
                id: p.id,
                title: p.title,
                artist: p.artist,
                genre: p.genre,
                year: p.year,
                tempo: p.tempo,
                danceability: p.danceability,
                energy: p.energy,
                valence: p.valence || 0.5,
                acousticness: p.acousticness || 0.2,
                instrumentalness: p.instrumentalness || 0.0,
                speechiness: p.speechiness || 0.05,
                pca_x: p.x,
                pca_y: p.y,
                feature_vector: p.feature_vector || [
                    p.danceability, p.energy, p.valence || 0.5,
                    (p.tempo - 120) / 30, p.acousticness || 0.2,
                    p.instrumentalness || 0.0, p.speechiness || 0.05, -6.0
                ]
            }));
        }

        setupEventListeners();
        renderLegend();
        renderCanvas();
        renderRadarChart();
        startAnimationLoop();
        updateSilhouetteStatus(state.metrics.silhouette);
    }

    // Dynamic Coordinate Bounds (ensures no song or centroid is clipped)
    function getCanvasBounds() {
        let minX = -8.0, maxX = 3.0, minY = -2.8, maxY = 3.2;

        if (state.points && state.points.length > 0) {
            let pMinX = Infinity, pMaxX = -Infinity, pMinY = Infinity, pMaxY = -Infinity;

            for (let i = 0; i < state.points.length; i++) {
                const pt = state.points[i];
                if (pt.x < pMinX) pMinX = pt.x;
                if (pt.x > pMaxX) pMaxX = pt.x;
                if (pt.y < pMinY) pMinY = pt.y;
                if (pt.y > pMaxY) pMaxY = pt.y;
            }

            if (state.seedSong && typeof state.seedSong.pca_x === "number") {
                if (state.seedSong.pca_x < pMinX) pMinX = state.seedSong.pca_x;
                if (state.seedSong.pca_x > pMaxX) pMaxX = state.seedSong.pca_x;
                if (state.seedSong.pca_y < pMinY) pMinY = state.seedSong.pca_y;
                if (state.seedSong.pca_y > pMaxY) pMaxY = state.seedSong.pca_y;
            }

            if (state.centroids && state.centroids.length > 0) {
                state.centroids.forEach(c => {
                    if (c && c.length >= 2) {
                        if (c[0] < pMinX) pMinX = c[0];
                        if (c[0] > pMaxX) pMaxX = c[0];
                        if (c[1] < pMinY) pMinY = c[1];
                        if (c[1] > pMaxY) pMaxY = c[1];
                    }
                });
            }

            const spanX = (pMaxX - pMinX) || 1.0;
            const spanY = (pMaxY - pMinY) || 1.0;
            minX = pMinX - spanX * 0.08;
            maxX = pMaxX + spanX * 0.08;
            minY = pMinY - spanY * 0.08;
            maxY = pMaxY + spanY * 0.08;
        }

        return { minX, maxX, minY, maxY };
    }

    // Coordinate Mapping (2D PCA -> Logical CSS Pixels)
    function mapCoords(x, y, width, height, bounds) {
        const padX = 42;
        const padY = 36;
        const px = padX + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (width - 2 * padX);
        const py = height - (padY + ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (height - 2 * padY));
        return { px, py };
    }

    // Synchronize High-DPI Canvas Buffer with Element Size
    function syncCanvasDPI(canvas) {
        if (!canvas) return { width: 800, height: 420, dpr: 1 };
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(300, Math.round(rect.width || canvas.clientWidth || 800));
        const height = Math.max(250, Math.round(rect.height || canvas.clientHeight || 420));
        const bufW = Math.round(width * dpr);
        const bufH = Math.round(height * dpr);

        if (canvas.width !== bufW || canvas.height !== bufH) {
            canvas.width = bufW;
            canvas.height = bufH;
        }
        return { width, height, dpr };
    }

    // Dynamic Cluster Color Legend
    function renderLegend() {
        if (!elements.canvasLegend) return;
        const nClusters = state.metrics.n_clusters || state.k;
        let html = "";

        const moodNames = [
            "Energetic Pop", "Workout / EDM", "Acoustic Chill", "Melodic R&B",
            "Alternative Rock", "Classic / Instrumental", "Latin Dance", "Ambient Space"
        ];

        for (let i = 0; i < nClusters; i++) {
            const color = state.clusterColors[i % state.clusterColors.length] || "#1DB954";
            const name = moodNames[i] || `Cluster ${i + 1}`;
            html += `
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: ${color}"></span>
                    <span>${name}</span>
                </div>
            `;
        }

        if (state.algorithm === "dbscan" && state.metrics.n_noise > 0) {
            html += `
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #64748B"></span>
                    <span>Noise / Outliers</span>
                </div>
            `;
        }

        elements.canvasLegend.innerHTML = html;
    }

    // Animation Loop for Pulsing Glow on Active Seed Song
    function startAnimationLoop() {
        function loop() {
            state.pulseAngle += 0.04;
            renderCanvas();
            state.animFrameId = requestAnimationFrame(loop);
        }
        loop();
    }

    // Main 2D Canvas Renderer
    function renderCanvas() {
        const canvas = elements.audioSpaceCanvas;
        if (!canvas) return;

        const { width, height, dpr } = syncCanvasDPI(canvas);
        const ctx = canvas.getContext("2d");

        ctx.save();
        ctx.scale(dpr, dpr);

        const bounds = getCanvasBounds();

        // 1. Clear Background
        ctx.fillStyle = "#070B12";
        ctx.fillRect(0, 0, width, height);

        // 2. Subtle Background Grid Lines
        ctx.strokeStyle = "rgba(27, 41, 69, 0.4)";
        ctx.lineWidth = 1;
        const gridStep = 45;
        for (let x = gridStep; x < width; x += gridStep) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = gridStep; y < height; y += gridStep) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // 3. Origin Crosshair (0, 0 in PCA space)
        if (bounds.minX <= 0 && bounds.maxX >= 0 && bounds.minY <= 0 && bounds.maxY >= 0) {
            const origin = mapCoords(0, 0, width, height, bounds);
            ctx.strokeStyle = "rgba(56, 189, 248, 0.22)";
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(origin.px, 0);
            ctx.lineTo(origin.px, height);
            ctx.moveTo(0, origin.py);
            ctx.lineTo(width, origin.py);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 4. Coordinates for Active Seed Song
        let seedCoords = null;
        if (state.seedSong && typeof state.seedSong.pca_x === "number") {
            seedCoords = mapCoords(state.seedSong.pca_x, state.seedSong.pca_y, width, height, bounds);
        }

        // 5. Draw Connecting Rays from Seed Track to Top Recommendations
        if (seedCoords && state.recommendations && state.recommendations.length > 0) {
            state.recommendations.forEach((rec, idx) => {
                if (typeof rec.pca_x !== "number") return;
                const recCoords = mapCoords(rec.pca_x, rec.pca_y, width, height, bounds);

                const grad = ctx.createLinearGradient(seedCoords.px, seedCoords.py, recCoords.px, recCoords.py);
                grad.addColorStop(0, "rgba(29, 185, 84, 0.75)");
                grad.addColorStop(1, "rgba(56, 189, 248, 0.35)");

                ctx.strokeStyle = grad;
                ctx.lineWidth = Math.max(1.2, 2.8 - idx * 0.35);
                ctx.setLineDash([5, 3]);
                ctx.beginPath();
                ctx.moveTo(seedCoords.px, seedCoords.py);
                ctx.lineTo(recCoords.px, recCoords.py);
                ctx.stroke();
                ctx.setLineDash([]);

                // Midpoint Rank Badge
                const midX = (seedCoords.px + recCoords.px) / 2;
                const midY = (seedCoords.py + recCoords.py) / 2;
                ctx.fillStyle = "rgba(13, 20, 36, 0.88)";
                ctx.fillRect(midX - 14, midY - 8, 28, 16);
                ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
                ctx.strokeRect(midX - 14, midY - 8, 28, 16);
                ctx.fillStyle = "#38BDF8";
                ctx.font = "9px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(`#${rec.rank}`, midX, midY);
            });
        }

        // 6. Draw Centroids
        if (state.centroids && state.centroids.length > 0) {
            state.centroids.forEach((c, idx) => {
                if (!c || c.length < 2) return;
                const cPos = mapCoords(c[0], c[1], width, height, bounds);
                const color = state.clusterColors[idx % state.clusterColors.length] || "#1DB954";

                // Centroid outer halo
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(cPos.px, cPos.py, 13, 0, Math.PI * 2);
                ctx.stroke();

                // Crosshair
                ctx.beginPath();
                ctx.moveTo(cPos.px - 16, cPos.py);
                ctx.lineTo(cPos.px + 16, cPos.py);
                ctx.moveTo(cPos.px, cPos.py - 16);
                ctx.lineTo(cPos.px, cPos.py + 16);
                ctx.stroke();

                // Center Dot
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(cPos.px, cPos.py, 4.5, 0, Math.PI * 2);
                ctx.fill();

                // Label
                ctx.fillStyle = "#F8FAFC";
                ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(`C${idx + 1}`, cPos.px, cPos.py - 17);
            });
        }

        // 7. Draw Song Points
        state.points.forEach((p) => {
            const pos = mapCoords(p.x, p.y, width, height, bounds);
            const isHovered = state.hoveredPoint && state.hoveredPoint.id === p.id;
            const isSeed = state.seedSong && state.seedSong.id === p.id;

            let radius = isHovered ? 7.5 : (p.type === "noise" ? 3.5 : 4.5);
            let color = p.color || "#1DB954";

            // Highlight if point is one of the recommendations
            const isRec = state.recommendations.some(r => r.id === p.id);
            if (isRec) {
                radius = isHovered ? 8.5 : 6.5;
            }

            // Glow around point
            if (isHovered || isRec) {
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.35;
                ctx.beginPath();
                ctx.arc(pos.px, pos.py, radius + 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            // Core point circle
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, radius, 0, Math.PI * 2);
            ctx.fill();

            // Border stroke
            ctx.strokeStyle = isHovered ? "#FFFFFF" : (isRec ? "#38BDF8" : "rgba(0,0,0,0.5)");
            ctx.lineWidth = isHovered ? 2 : (isRec ? 1.5 : 0.8);
            ctx.stroke();
        });

        // 8. Draw Active Seed Song Pulse Indicator
        if (seedCoords) {
            const pulseRadius = 14 + Math.sin(state.pulseAngle) * 4;
            const pulseAlpha = 0.4 + Math.sin(state.pulseAngle) * 0.25;

            // Animated neon pulse ring
            ctx.strokeStyle = `rgba(29, 185, 84, ${pulseAlpha})`;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(seedCoords.px, seedCoords.py, pulseRadius, 0, Math.PI * 2);
            ctx.stroke();

            // Inner solid seed marker
            ctx.fillStyle = "#1DB954";
            ctx.beginPath();
            ctx.arc(seedCoords.px, seedCoords.py, 8, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Text Label
            ctx.fillStyle = "#1ED760";
            ctx.font = "bold 10px 'Plus Jakarta Sans', sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("★ SEED", seedCoords.px, seedCoords.py - 18);
        }

        ctx.restore();
    }

    // Custom Lightweight Radar Spider Chart
    function renderRadarChart() {
        const canvas = elements.radarCanvas;
        if (!canvas) return;

        const dpr = window.devicePixelRatio || 1;
        const width = 340;
        const height = 210;
        canvas.width = width * dpr;
        canvas.height = height * dpr;

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, width, height);

        const centerX = width / 2;
        const centerY = height / 2 + 10;
        const radius = 68;

        const radarData = state.radar || {
            labels: ["Dance", "Energy", "Valence", "Acoustic", "Instrumental", "Speech"],
            seed_values: [0.6, 0.7, 0.5, 0.1, 0.0, 0.05],
            rec_values: [0.65, 0.75, 0.55, 0.12, 0.0, 0.06]
        };

        const numAxes = radarData.labels.length;
        const angleStep = (Math.PI * 2) / numAxes;

        // Draw Web / Concentric Polygons
        const levels = 4;
        for (let l = 1; l <= levels; l++) {
            const levelRadius = (radius / levels) * l;
            ctx.strokeStyle = "rgba(27, 41, 69, 0.6)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < numAxes; i++) {
                const angle = i * angleStep - Math.PI / 2;
                const x = centerX + Math.cos(angle) * levelRadius;
                const y = centerY + Math.sin(angle) * levelRadius;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
        }

        // Draw Radial Spokes & Dimension Labels
        ctx.font = "9.5px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#94A3B8";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (let i = 0; i < numAxes; i++) {
            const angle = i * angleStep - Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;

            ctx.strokeStyle = "rgba(27, 41, 69, 0.7)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(x, y);
            ctx.stroke();

            const labelX = centerX + Math.cos(angle) * (radius + 18);
            const labelY = centerY + Math.sin(angle) * (radius + 18);
            ctx.fillText(radarData.labels[i], labelX, labelY);
        }

        // Draw Top Match Recommendation Polygon (Sky Blue)
        if (radarData.rec_values && radarData.rec_values.length === numAxes) {
            ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
            ctx.strokeStyle = "#38BDF8";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < numAxes; i++) {
                const val = Math.max(0, Math.min(1, radarData.rec_values[i]));
                const angle = i * angleStep - Math.PI / 2;
                const x = centerX + Math.cos(angle) * (radius * val);
                const y = centerY + Math.sin(angle) * (radius * val);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }

        // Draw Seed Song Polygon (Spotify Green)
        if (radarData.seed_values && radarData.seed_values.length === numAxes) {
            ctx.fillStyle = "rgba(29, 185, 84, 0.35)";
            ctx.strokeStyle = "#1DB954";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < numAxes; i++) {
                const val = Math.max(0, Math.min(1, radarData.seed_values[i]));
                const angle = i * angleStep - Math.PI / 2;
                const x = centerX + Math.cos(angle) * (radius * val);
                const y = centerY + Math.sin(angle) * (radius * val);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }

    // Silhouette Status Evaluation Badge
    function updateSilhouetteStatus(score) {
        if (!elements.metricSilBadge) return;
        if (score === "N/A" || score === null || score === undefined) {
            elements.metricSilBadge.textContent = "N/A";
            elements.metricSilBadge.style.color = "#64748B";
            return;
        }
        const s = parseFloat(score);
        if (s >= 0.35) {
            elements.metricSilBadge.textContent = "✨ Strong Structure";
            elements.metricSilBadge.style.color = "#1DB954";
        } else if (s >= 0.20) {
            elements.metricSilBadge.textContent = "✨ Reasonable";
            elements.metricSilBadge.style.color = "#38BDF8";
        } else if (s >= 0.05) {
            elements.metricSilBadge.textContent = "⚡ Overlapping";
            elements.metricSilBadge.style.color = "#F59E0B";
        } else {
            elements.metricSilBadge.textContent = "⚠️ Dispersed";
            elements.metricSilBadge.style.color = "#F43F5E";
        }
    }

    // Update Seed Track Banner and Profile Card
    function updateSeedBanner(seed) {
        if (!seed) return;
        if (elements.bannerTitle) elements.bannerTitle.textContent = seed.title;
        if (elements.bannerArtist) elements.bannerArtist.textContent = `${seed.artist} (${seed.year || 2024})`;
        if (elements.bannerGenre) elements.bannerGenre.textContent = `🏷️ ${seed.genre}`;
        if (elements.bannerTempo) elements.bannerTempo.textContent = `🥁 ${seed.tempo} BPM`;

        // Profile Card
        if (elements.profileSeedTitle) elements.profileSeedTitle.textContent = seed.title;
        if (elements.profileSeedArtist) elements.profileSeedArtist.textContent = seed.artist;
        if (elements.barSeedDance) elements.barSeedDance.style.width = `${seed.danceability * 100}%`;
        if (elements.valSeedDance) elements.valSeedDance.textContent = seed.danceability;
        if (elements.barSeedEnergy) elements.barSeedEnergy.style.width = `${seed.energy * 100}%`;
        if (elements.valSeedEnergy) elements.valSeedEnergy.textContent = seed.energy;
        if (elements.barSeedValence) elements.barSeedValence.style.width = `${seed.valence * 100}%`;
        if (elements.valSeedValence) elements.valSeedValence.textContent = seed.valence;
        if (elements.barSeedAcoustic) elements.barSeedAcoustic.style.width = `${seed.acousticness * 100}%`;
        if (elements.valSeedAcoustic) elements.valSeedAcoustic.textContent = seed.acousticness;
    }

    // Update UI Elements with State
    function updateUI(data) {
        if (data.metrics) {
            state.metrics = data.metrics;
            if (elements.metricClusters) elements.metricClusters.textContent = data.metrics.n_clusters;
            if (elements.metricSilhouette) elements.metricSilhouette.textContent = data.metrics.silhouette;
            if (elements.metricInertia) elements.metricInertia.textContent = data.metrics.inertia;
            if (elements.metricNoise) elements.metricNoise.textContent = data.metrics.n_noise;
            updateSilhouetteStatus(data.metrics.silhouette);
        }

        if (data.points) state.points = data.points;
        if (data.centroids) state.centroids = data.centroids;
        if (data.seed_song) state.seedSong = data.seed_song;
        if (data.recommendations) state.recommendations = data.recommendations;
        if (data.radar) state.radar = data.radar;

        // Update Seed Track Banner & Profile
        if (state.seedSong) {
            updateSeedBanner(state.seedSong);

            if (elements.songSelect && state.seedSong.id !== "custom") {
                elements.songSelect.value = state.seedSong.id;
            }
        }

        // Update Recommendations Table
        if (state.recommendations && elements.recsGrid) {
            if (elements.recCountBadge) {
                elements.recCountBadge.textContent = `Top ${state.recommendations.length}`;
            }

            let recsHtml = "";
            state.recommendations.forEach(rec => {
                recsHtml += `
                    <div class="playlist-row" data-song-id="${rec.id}">
                        <div class="row-rank">#${rec.rank}</div>
                        <div class="row-info">
                            <span class="row-title">${rec.title}</span>
                            <span class="row-artist">${rec.artist}</span>
                        </div>
                        <div class="row-tags">
                            <span class="tag-pill">${rec.genre}</span>
                            <span class="tag-pill">🥁 ${rec.tempo} BPM</span>
                        </div>
                        <div class="row-match">
                            <div class="match-bar-track">
                                <div class="match-bar-fill" style="width: ${rec.match_percentage}%"></div>
                            </div>
                            <span class="match-text">${rec.match_percentage}%</span>
                        </div>
                        <button class="btn-play-seed" data-song-id="${rec.id}" title="Select as Seed Track">
                            ▶ Select
                        </button>
                    </div>
                `;
            });
            elements.recsGrid.innerHTML = recsHtml;

            // Attach listeners to "Select" buttons in table
            document.querySelectorAll(".btn-play-seed").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const songId = btn.getAttribute("data-song-id");
                    selectSeedSong(songId);
                });
            });
        }

        // Update Radar Legends
        if (state.radar) {
            if (elements.radarSeedLegend) elements.radarSeedLegend.textContent = state.radar.seed_title || "Seed";
            if (elements.radarRecLegend) elements.radarRecLegend.textContent = state.radar.rec_title || "Top Match";
        }

        // Update Latency & Insight
        if (data.metrics && data.metrics.inference_time_ms !== undefined && elements.recLatency) {
            elements.recLatency.textContent = `Latency: ~${data.metrics.inference_time_ms} ms`;
        }
        if (data.explanation && elements.explanationText) {
            elements.explanationText.textContent = data.explanation;
        }

        renderLegend();
        renderCanvas();
        renderRadarChart();
    }

    // Trigger Clustering & Recommendations (Instant Client Execution + Server Sync)
    function triggerClusterUpdate() {
        const t0 = performance.now();

        // Ensure we have active tracks
        if (state.tracks.length > 0) {
            let clusterResult;
            let explanation = "";

            if (state.algorithm === "kmeans") {
                clusterResult = runKMeansClient(state.tracks, state.k, state.init);
                explanation = `🎯 K-Means: Partitioned ${state.tracks.length} tracks into K=${state.k} acoustic clusters (SSE = ${clusterResult.sse}). '${state.init}' initialization provides well-distributed centroids.`;
            } else if (state.algorithm === "hierarchical") {
                clusterResult = runHierarchicalClient(state.tracks, state.k, state.linkage);
                explanation = `🌿 Hierarchical: Built agglomerative tree (${state.linkage} linkage) with ${state.k} clusters capturing musical genre proximity.`;
            } else if (state.algorithm === "dbscan") {
                clusterResult = runDBSCANClient(state.tracks, state.eps, state.minSamples);
                explanation = `⁘ DBSCAN: Discovered ${clusterResult.nClusters} natural density clusters (ε=${state.eps.toFixed(2)}, MinPts=${state.minSamples}) with ${clusterResult.nNoise} noise outliers.`;
            }

            const silScore = calcSilhouetteClient(state.tracks, clusterResult.labels);

            // Update points payload
            const pointsPayload = state.tracks.map((t, i) => {
                const cId = clusterResult.labels[i];
                const color = cId === -1 ? "#64748B" : state.clusterColors[cId % state.clusterColors.length];
                const pType = cId === -1 ? "noise" : "normal";
                return {
                    id: t.id,
                    title: t.title,
                    artist: t.artist,
                    genre: t.genre,
                    year: t.year,
                    tempo: t.tempo,
                    danceability: t.danceability,
                    energy: t.energy,
                    valence: t.valence,
                    acousticness: t.acousticness,
                    instrumentalness: t.instrumentalness,
                    speechiness: t.speechiness,
                    x: t.pca_x,
                    y: t.pca_y,
                    cluster: cId,
                    type: pType,
                    color: color
                };
            });

            // Active seed song
            const activeSeed = state.seedSong || state.tracks[0];

            // Recommendations
            const { recommendations, radar } = generateRecommendationsClient(
                activeSeed,
                state.tracks,
                clusterResult.labels,
                state.metric,
                state.topN,
                state.filterCluster
            );

            const elapsedMs = Math.round((performance.now() - t0) * 10) / 10;

            const clientMetrics = {
                n_clusters: clusterResult.nClusters || state.k,
                n_noise: clusterResult.nNoise || 0,
                silhouette: silScore,
                inertia: clusterResult.sse,
                inference_time_ms: elapsedMs
            };

            // Instant UI update
            updateUI({
                metrics: clientMetrics,
                points: pointsPayload,
                centroids: clusterResult.centroids2D,
                seed_song: activeSeed,
                recommendations: recommendations,
                radar: radar,
                explanation: explanation
            });
        }

        // Asynchronous Background Server Sync (if backend is active)
        const payload = {
            algorithm: state.algorithm,
            k: state.k,
            init: state.init,
            linkage: state.linkage,
            eps: state.eps,
            min_samples: state.minSamples,
            seed_id: state.seedId,
            metric: state.metric,
            top_n: state.topN,
            filter_cluster: state.filterCluster,
            custom_features: state.customFeatures
        };

        fetch("/api/cluster", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) throw new Error("Server status " + res.status);
            return res.json();
        })
        .then(data => {
            if (data && data.success) {
                updateUI(data);
            }
        })
        .catch(err => {
            // Client ML engine already handled the update smoothly
            console.debug("Backend API offline; running client ML engine:", err.message);
        });
    }

    // Select a New Seed Song with Instant Reactive Visual Feedback
    function selectSeedSong(songId) {
        state.seedId = songId;
        state.customFeatures = null;

        const found = state.tracks.find(t => t.id === songId) || state.points.find(p => p.id === songId);
        if (found) {
            state.seedSong = {
                id: found.id,
                title: found.title,
                artist: found.artist,
                genre: found.genre,
                year: found.year,
                tempo: found.tempo,
                danceability: found.danceability,
                energy: found.energy,
                valence: found.valence || 0.5,
                acousticness: found.acousticness || 0.2,
                instrumentalness: found.instrumentalness || 0.0,
                speechiness: found.speechiness || 0.05,
                feature_vector: found.feature_vector || [
                    found.danceability, found.energy, found.valence || 0.5,
                    (found.tempo - 120) / 30, found.acousticness || 0.2,
                    found.instrumentalness || 0.0, found.speechiness || 0.05, -6.0
                ],
                pca_x: found.x !== undefined ? found.x : found.pca_x,
                pca_y: found.y !== undefined ? found.y : found.pca_y
            };
            if (elements.songSelect && elements.songSelect.value !== songId) {
                elements.songSelect.value = songId;
            }
            updateSeedBanner(state.seedSong);
            renderCanvas();
        }

        triggerClusterUpdate();
    }

    // Setup All Event Listeners
    function setupEventListeners() {
        // 1. Dropdown Selection
        if (elements.songSelect) {
            elements.songSelect.addEventListener("change", (e) => {
                selectSeedSong(e.target.value);
            });
        }

        // 2. Random Seed Song Button
        if (elements.randomSongBtn && elements.songSelect) {
            elements.randomSongBtn.addEventListener("click", () => {
                const options = elements.songSelect.options;
                if (!options || options.length === 0) return;
                const randomIndex = Math.floor(Math.random() * options.length);
                elements.songSelect.selectedIndex = randomIndex;
                selectSeedSong(options[randomIndex].value);
            });
        }

        // 3. Algorithm Tabs
        elements.algoTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                elements.algoTabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                state.algorithm = tab.getAttribute("data-algo");

                // Toggle visibility
                if (elements.kmeansControls) elements.kmeansControls.classList.toggle("hidden", state.algorithm !== "kmeans");
                if (elements.hierarchicalControls) elements.hierarchicalControls.classList.toggle("hidden", state.algorithm !== "hierarchical");
                if (elements.dbscanControls) elements.dbscanControls.classList.toggle("hidden", state.algorithm !== "dbscan");

                triggerClusterUpdate();
            });
        });

        // 4. K-Means Sliders
        if (elements.kSlider) {
            elements.kSlider.addEventListener("input", (e) => {
                state.k = parseInt(e.target.value);
                if (elements.kVal) elements.kVal.textContent = state.k;
                triggerClusterUpdate();
            });
        }

        elements.initBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.initBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.init = btn.getAttribute("data-init");
                triggerClusterUpdate();
            });
        });

        // 5. Hierarchical Controls
        if (elements.hierKSlider) {
            elements.hierKSlider.addEventListener("input", (e) => {
                state.k = parseInt(e.target.value);
                if (elements.hierKVal) elements.hierKVal.textContent = state.k;
                triggerClusterUpdate();
            });
        }

        if (elements.linkageSelect) {
            elements.linkageSelect.addEventListener("change", (e) => {
                state.linkage = e.target.value;
                triggerClusterUpdate();
            });
        }

        // 6. DBSCAN Controls
        if (elements.epsSlider) {
            elements.epsSlider.addEventListener("input", (e) => {
                state.eps = parseFloat(e.target.value);
                if (elements.epsVal) elements.epsVal.textContent = state.eps.toFixed(2);
                debounce(triggerClusterUpdate, 100);
            });
        }

        if (elements.minPtsSlider) {
            elements.minPtsSlider.addEventListener("input", (e) => {
                state.minSamples = parseInt(e.target.value);
                if (elements.minPtsVal) elements.minPtsVal.textContent = state.minSamples;
                debounce(triggerClusterUpdate, 100);
            });
        }

        // 7. Distance Metric & Filter
        elements.metricBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.metricBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.metric = btn.getAttribute("data-metric");
                triggerClusterUpdate();
            });
        });

        if (elements.filterClusterCheckbox) {
            elements.filterClusterCheckbox.addEventListener("change", (e) => {
                state.filterCluster = e.target.checked;
                triggerClusterUpdate();
            });
        }

        if (elements.topNSlider) {
            elements.topNSlider.addEventListener("input", (e) => {
                state.topN = parseInt(e.target.value);
                if (elements.topNVal) elements.topNVal.textContent = state.topN;
                triggerClusterUpdate();
            });
        }

        // 8. Custom Audio Sliders
        const updateSliderDisplay = (slider, display) => {
            if (slider && display) {
                slider.addEventListener("input", (e) => {
                    display.textContent = e.target.value;
                });
            }
        };
        updateSliderDisplay(elements.sliderDance, elements.sliderDanceVal);
        updateSliderDisplay(elements.sliderEnergy, elements.sliderEnergyVal);
        updateSliderDisplay(elements.sliderValence, elements.sliderValenceVal);
        updateSliderDisplay(elements.sliderTempo, elements.sliderTempoVal);
        updateSliderDisplay(elements.sliderAcoustic, elements.sliderAcousticVal);

        if (elements.applyCustomBtn) {
            elements.applyCustomBtn.addEventListener("click", () => {
                const dance = parseFloat(elements.sliderDance.value);
                const energy = parseFloat(elements.sliderEnergy.value);
                const valence = parseFloat(elements.sliderValence.value);
                const tempo = parseFloat(elements.sliderTempo.value);
                const acoustic = parseFloat(elements.sliderAcoustic.value);

                state.customFeatures = [dance, energy, valence, tempo, acoustic, 0.0, 0.06, -6.0];

                // Approximate 2D PCA projection for custom tuned vector
                const customPcaX = Number(((dance - 0.6) * 1.8 + (tempo - 120) * 0.02 - (acoustic - 0.3) * 2.2).toFixed(4));
                const customPcaY = Number(((energy - 0.6) * 2.5 - (valence - 0.5) * 1.5).toFixed(4));

                state.seedSong = {
                    id: "custom",
                    title: "Custom Tuned Track",
                    artist: "User Profile",
                    genre: "Custom Audio",
                    year: 2026,
                    danceability: dance,
                    energy: energy,
                    valence: valence,
                    tempo: tempo,
                    acousticness: acoustic,
                    instrumentalness: 0.0,
                    speechiness: 0.06,
                    pca_x: customPcaX,
                    pca_y: customPcaY,
                    feature_vector: [
                        (dance - 0.65) / 0.15,
                        (energy - 0.65) / 0.18,
                        (valence - 0.5) / 0.22,
                        (tempo - 120) / 28,
                        (acoustic - 0.25) / 0.3,
                        -0.3,
                        0.0,
                        0.2
                    ]
                };

                updateSeedBanner(state.seedSong);
                triggerClusterUpdate();
            });
        }

        // 9. Interactive Canvas Hover & Click (Point Selection)
        const canvas = elements.audioSpaceCanvas;
        if (canvas) {
            canvas.addEventListener("mousemove", (e) => {
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                const width = rect.width;
                const height = rect.height;
                const bounds = getCanvasBounds();

                let closestPoint = null;
                let minDist = 14;

                state.points.forEach(p => {
                    const pos = mapCoords(p.x, p.y, width, height, bounds);
                    const dx = pos.px - mouseX;
                    const dy = pos.py - mouseY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDist) {
                        minDist = dist;
                        closestPoint = p;
                    }
                });

                state.hoveredPoint = closestPoint;

                if (closestPoint && elements.canvasTooltip) {
                    elements.canvasTooltip.style.display = "block";
                    elements.canvasTooltip.style.left = `${mouseX + 15}px`;
                    elements.canvasTooltip.style.top = `${mouseY - 20}px`;

                    const moodName = closestPoint.cluster === -1 ? "Noise / Outlier" : `Cluster #${closestPoint.cluster + 1}`;
                    elements.canvasTooltip.innerHTML = `
                        <div class="tooltip-title">${closestPoint.title}</div>
                        <div class="tooltip-artist">${closestPoint.artist} • ${closestPoint.genre}</div>
                        <div class="tooltip-metrics">
                            <span class="tooltip-badge">🥁 ${closestPoint.tempo} BPM</span>
                            <span class="tooltip-badge">⚡ ${closestPoint.energy} Energy</span>
                            <span class="tooltip-badge">💃 ${closestPoint.danceability} Dance</span>
                            <span class="tooltip-badge" style="color:${closestPoint.color}">${moodName}</span>
                        </div>
                    `;
                } else if (elements.canvasTooltip) {
                    elements.canvasTooltip.style.display = "none";
                }
            });

            canvas.addEventListener("mouseleave", () => {
                state.hoveredPoint = null;
                if (elements.canvasTooltip) elements.canvasTooltip.style.display = "none";
            });

            canvas.addEventListener("click", () => {
                if (state.hoveredPoint) {
                    selectSeedSong(state.hoveredPoint.id);
                }
            });
        }

        // 10. Validation Curves Modal Triggers
        const openModal = () => {
            if (elements.curveModal) elements.curveModal.classList.remove("hidden");
            loadValidationCurves();
        };

        const closeModal = () => {
            if (elements.curveModal) elements.curveModal.classList.add("hidden");
        };

        if (elements.openCurveBtn) elements.openCurveBtn.addEventListener("click", openModal);
        if (elements.closeCurveBtn) elements.closeCurveBtn.addEventListener("click", closeModal);
        if (elements.curveModal) {
            elements.curveModal.addEventListener("click", (e) => {
                if (e.target === elements.curveModal) closeModal();
            });
        }

        // Responsive window redraw
        window.addEventListener("resize", () => {
            renderCanvas();
        });
    }

    // Debounce Helper
    function debounce(func, wait) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(func, wait);
    }

    // Load and Render Validation Curves (Elbow SSE & Silhouette)
    function loadValidationCurves() {
        if (state.validationCurves) {
            renderValidationCharts(state.validationCurves);
            return;
        }

        if (elements.modalBestKText) {
            elements.modalBestKText.textContent = "Calculating validation curves across K in [2, 8]...";
        }

        // Compute immediately on client
        if (state.tracks.length > 0) {
            const kValues = [2, 3, 4, 5, 6, 7, 8];
            const elbowCurve = [];
            const silCurve = [];

            kValues.forEach(k => {
                const res = runKMeansClient(state.tracks, k, "k-means++");
                elbowCurve.push(res.sse);
                const s = calcSilhouetteClient(state.tracks, res.labels);
                silCurve.push(typeof s === "number" ? s : 0);
            });

            let bestK = 2;
            let bestSil = -1;
            kValues.forEach((k, idx) => {
                if (silCurve[idx] > bestSil) {
                    bestSil = silCurve[idx];
                    bestK = k;
                }
            });

            state.validationCurves = {
                k_values: kValues,
                elbow_curve: elbowCurve,
                silhouette_curve: silCurve,
                best_k: bestK
            };

            renderValidationCharts(state.validationCurves);
        }

        // Sync with backend if available
        fetch("/api/validation-curve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        })
        .then(res => res.json())
        .then(data => {
            if (data && data.success) {
                state.validationCurves = data;
                renderValidationCharts(data);
            }
        })
        .catch(err => console.debug("Validation curve client computed:", err));
    }

    function renderValidationCharts(data) {
        if (elements.modalBestKText) {
            const peakSil = Math.max(...data.silhouette_curve);
            elements.modalBestKText.innerHTML = `
                Recommended optimal clusters: <strong>K = ${data.best_k}</strong> 
                (Peak Silhouette Score: <strong>${peakSil}</strong>).
            `;
        }

        // 1. Elbow Chart (Chart.js)
        if (elements.elbowChartCanvas && window.Chart) {
            if (elbowChartInstance) elbowChartInstance.destroy();
            const ctxElbow = elements.elbowChartCanvas.getContext("2d");
            elbowChartInstance = new Chart(ctxElbow, {
                type: "line",
                data: {
                    labels: data.k_values.map(k => `K=${k}`),
                    datasets: [{
                        label: "Inertia (SSE)",
                        data: data.elbow_curve,
                        borderColor: "#F59E0B",
                        backgroundColor: "rgba(245, 158, 11, 0.15)",
                        borderWidth: 2.5,
                        pointBackgroundColor: "#F59E0B",
                        pointRadius: 4,
                        fill: true,
                        tension: 0.25
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { grid: { color: "rgba(27, 41, 69, 0.5)" }, ticks: { color: "#94A3B8" } },
                        y: { grid: { color: "rgba(27, 41, 69, 0.5)" }, ticks: { color: "#94A3B8" } }
                    }
                }
            });
        }

        // 2. Silhouette Chart (Chart.js)
        if (elements.silhouetteChartCanvas && window.Chart) {
            if (silChartInstance) silChartInstance.destroy();
            const ctxSil = elements.silhouetteChartCanvas.getContext("2d");
            silChartInstance = new Chart(ctxSil, {
                type: "line",
                data: {
                    labels: data.k_values.map(k => `K=${k}`),
                    datasets: [{
                        label: "Silhouette Score",
                        data: data.silhouette_curve,
                        borderColor: "#38BDF8",
                        backgroundColor: "rgba(56, 189, 248, 0.15)",
                        borderWidth: 2.5,
                        pointBackgroundColor: data.k_values.map(k => k === data.best_k ? "#1DB954" : "#38BDF8"),
                        pointRadius: data.k_values.map(k => k === data.best_k ? 8 : 4),
                        fill: true,
                        tension: 0.25
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { grid: { color: "rgba(27, 41, 69, 0.5)" }, ticks: { color: "#94A3B8" } },
                        y: { grid: { color: "rgba(27, 41, 69, 0.5)" }, ticks: { color: "#94A3B8" } }
                    }
                }
            });
        }
    }

    // Start on DOM Ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
