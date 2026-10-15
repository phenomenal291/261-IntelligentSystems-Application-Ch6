/**
 * Spotify Song Recommendation Studio - Client Controller
 * HK261 Intelligent Systems - Chapter 6 Application
 * Features: 2D Audio Feature Space Canvas, Radar Spider Visualizer, Interactive Clustering & Nearest Neighbor
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
        points: [],
        centroids: [],
        metrics: {},
        seedSong: null,
        recommendations: [],
        radar: null,
        clusterColors: [],
        hoveredPoint: null,
        animFrameId: null,
        pulseAngle: 0,
        validationCurves: null
    };

    // DOM Elements
    const elements = {
        // Dropdown & Banner
        songSelect: document.getElementById("songSelect"),
        randomSongBtn: document.getElementById("randomSongBtn"),
        bannerTitle: document.getElementById("bannerTitle"),
        bannerArtist: document.getElementById("bannerArtist"),
        bannerGenre: document.getElementById("bannerGenre"),
        bannerTempo: document.getElementById("bannerTempo"),
        samplesScroll: document.getElementById("samplesScroll"),

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

        // Algorithm Controls
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

        // Recommendation Settings
        metricBtns: document.querySelectorAll("[data-metric]"),
        filterClusterCheckbox: document.getElementById("filterClusterCheckbox"),
        topNSlider: document.getElementById("topNSlider"),
        topNVal: document.getElementById("topNVal"),

        // KPI Badges
        metricClusters: document.getElementById("metricClusters"),
        metricSilhouette: document.getElementById("metricSilhouette"),
        metricSilBadge: document.getElementById("metricSilBadge"),
        metricInertia: document.getElementById("metricInertia"),
        metricDB: document.getElementById("metricDB"),
        metricNoise: document.getElementById("metricNoise"),

        // Canvas & Tooltip
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

        // Modal
        curveModal: document.getElementById("curveModal"),
        openCurveBtnHeader: document.getElementById("openCurveBtnHeader"),
        openCurveBtnSidebar: document.getElementById("openCurveBtnSidebar"),
        closeCurveBtn: document.getElementById("closeCurveBtn"),
        elbowChartCanvas: document.getElementById("elbowChart"),
        silhouetteChartCanvas: document.getElementById("silhouetteChart"),
        modalBestKText: document.getElementById("modalBestKText")
    };

    let elbowChartInstance = null;
    let silChartInstance = null;
    let debounceTimer = null;

    // Initialize with Pre-computed Server Data
    function init() {
        if (window.INITIAL_DATA) {
            state.points = window.INITIAL_DATA.points || [];
            state.centroids = window.INITIAL_DATA.centroids || [];
            state.metrics = window.INITIAL_DATA.metrics || {};
            state.seedSong = window.INITIAL_DATA.seed_song || null;
            state.recommendations = window.INITIAL_DATA.recommendations || [];
            state.radar = window.INITIAL_DATA.radar || null;
            state.clusterColors = window.INITIAL_DATA.cluster_colors || [];

            if (state.seedSong) {
                state.seedId = state.seedSong.id;
            }
        }

        setupEventListeners();
        setupCanvasHighDPI();
        renderLegend();
        renderCanvas();
        renderRadarChart();
        startAnimationLoop();
        updateSilhouetteStatus(state.metrics.silhouette);
    }

    // High-DPI Canvas Setup
    function setupCanvasHighDPI() {
        const canvas = elements.audioSpaceCanvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = rect.width * dpr;
        canvas.height = 480 * dpr;

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
    }

    // Legend Generator
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
                    <span>Noise / Outlier Tracks</span>
                </div>
            `;
        }

        elements.canvasLegend.innerHTML = html;
    }

    // Coordinate Mapping (2D PCA Coordinates -> Canvas Pixels)
    function mapCoords(x, y, width, height) {
        // Bounds for 2D PCA in standard dataset: roughly [-3.5, 3.5]
        const padX = 50;
        const padY = 40;
        const minX = -3.8;
        const maxX = 3.8;
        const minY = -2.8;
        const maxY = 2.8;

        const px = padX + ((x - minX) / (maxX - minX)) * (width - 2 * padX);
        const py = height - (padY + ((y - minY) / (maxY - minY)) * (height - 2 * padY));
        return { px, py };
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
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.width / dpr;
        const height = canvas.height / dpr;
        const ctx = canvas.getContext("2d");

        // Clear Background
        ctx.fillStyle = "#080C14";
        ctx.fillRect(0, 0, width, height);

        // Subtle Background Grid Lines
        ctx.strokeStyle = "rgba(34, 48, 74, 0.4)";
        ctx.lineWidth = 1;
        const gridStep = 40;
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

        // Draw Origin Crosshair
        const origin = mapCoords(0, 0, width, height);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(origin.px, 0);
        ctx.lineTo(origin.px, height);
        ctx.moveTo(0, origin.py);
        ctx.lineTo(width, origin.py);
        ctx.stroke();
        ctx.setLineDash([]);

        // Active Seed Song Coordinates
        let seedCoords = null;
        if (state.seedSong) {
            seedCoords = mapCoords(state.seedSong.pca_x, state.seedSong.pca_y, width, height);
        }

        // Draw Connecting Rays from Seed Track to Top Recommendations
        if (seedCoords && state.recommendations && state.recommendations.length > 0) {
            state.recommendations.forEach((rec, idx) => {
                const recCoords = mapCoords(rec.pca_x, rec.pca_y, width, height);

                // Gradient ray
                const grad = ctx.createLinearGradient(seedCoords.px, seedCoords.py, recCoords.px, recCoords.py);
                grad.addColorStop(0, "rgba(29, 185, 84, 0.7)");
                grad.addColorStop(1, "rgba(56, 189, 248, 0.3)");

                ctx.strokeStyle = grad;
                ctx.lineWidth = Math.max(1, 3.0 - idx * 0.4);
                ctx.setLineDash([6, 3]);
                ctx.beginPath();
                ctx.moveTo(seedCoords.px, seedCoords.py);
                ctx.lineTo(recCoords.px, recCoords.py);
                ctx.stroke();
                ctx.setLineDash([]);

                // Draw distance pill label on ray midpoint
                const midX = (seedCoords.px + recCoords.px) / 2;
                const midY = (seedCoords.py + recCoords.py) / 2;
                ctx.fillStyle = "rgba(11, 15, 23, 0.85)";
                ctx.fillRect(midX - 16, midY - 8, 32, 16);
                ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
                ctx.strokeRect(midX - 16, midY - 8, 32, 16);
                ctx.fillStyle = "#38BDF8";
                ctx.font = "9px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(`#${rec.rank}`, midX, midY);
            });
        }

        // Draw Centroids
        if (state.centroids && state.centroids.length > 0) {
            state.centroids.forEach((c, idx) => {
                const cPos = mapCoords(c[0], c[1], width, height);
                const color = state.clusterColors[idx % state.clusterColors.length] || "#1DB954";

                // Centroid outer halo
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(cPos.px, cPos.py, 13, 0, Math.PI * 2);
                ctx.stroke();

                // Crosshair
                ctx.beginPath();
                ctx.moveTo(cPos.px - 17, cPos.py);
                ctx.lineTo(cPos.px + 17, cPos.py);
                ctx.moveTo(cPos.px, cPos.py - 17);
                ctx.lineTo(cPos.px, cPos.py + 17);
                ctx.stroke();

                // Centroid badge label
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(cPos.px, cPos.py, 5, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = "#F8FAFC";
                ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(`C${idx + 1}`, cPos.px, cPos.py - 18);
            });
        }

        // Draw Song Points
        state.points.forEach((p) => {
            const pos = mapCoords(p.x, p.y, width, height);
            const isHovered = state.hoveredPoint && state.hoveredPoint.id === p.id;
            const isSeed = state.seedSong && state.seedSong.id === p.id;

            let radius = isHovered ? 7 : (p.type === "noise" ? 3.5 : 4.5);
            let color = p.color;

            // Highlight if point is one of the recommendations
            const isRec = state.recommendations.some(r => r.id === p.id);
            if (isRec) {
                radius = isHovered ? 8 : 6.5;
            }

            // Glow around point
            if (isHovered || isRec) {
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.3;
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

        // Draw Active Seed Song Pulse Indicator
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
        const radius = 70;

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
            ctx.strokeStyle = "rgba(34, 48, 74, 0.6)";
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

        // Draw Axis Rays & Labels
        ctx.font = "10px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#94A3B8";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (let i = 0; i < numAxes; i++) {
            const angle = i * angleStep - Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;

            ctx.strokeStyle = "rgba(51, 69, 102, 0.5)";
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(x, y);
            ctx.stroke();

            // Label position slightly outside
            const labelX = centerX + Math.cos(angle) * (radius + 18);
            const labelY = centerY + Math.sin(angle) * (radius + 14);
            ctx.fillText(radarData.labels[i], labelX, labelY);
        }

        // Draw Recommended Song Polygon (Sky Blue)
        if (radarData.rec_values && radarData.rec_values.length === numAxes) {
            ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
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

    // Silhouette Status Evaluation Tag
    function updateSilhouetteStatus(score) {
        if (!elements.metricSilBadge) return;
        if (score === "N/A" || score === null || score === undefined) {
            elements.metricSilBadge.textContent = "Chưa đủ cụm";
            elements.metricSilBadge.style.color = "#64748B";
            return;
        }
        const s = parseFloat(score);
        if (s >= 0.35) {
            elements.metricSilBadge.textContent = "✨ Phân tách rất tốt";
            elements.metricSilBadge.style.color = "#1DB954";
        } else if (s >= 0.20) {
            elements.metricSilBadge.textContent = "✨ Cụm hợp lý";
            elements.metricSilBadge.style.color = "#38BDF8";
        } else if (s >= 0.05) {
            elements.metricSilBadge.textContent = "⚡ Có giao thoa nhẹ";
            elements.metricSilBadge.style.color = "#F59E0B";
        } else {
            elements.metricSilBadge.textContent = "⚠️ Cụm phân tán";
            elements.metricSilBadge.style.color = "#F43F5E";
        }
    }

    // Update UI Elements with Latest State
    function updateUI(data) {
        if (data.metrics) {
            state.metrics = data.metrics;
            elements.metricClusters.textContent = data.metrics.n_clusters;
            elements.metricSilhouette.textContent = data.metrics.silhouette;
            elements.metricInertia.textContent = data.metrics.inertia;
            elements.metricNoise.textContent = data.metrics.n_noise;
            updateSilhouetteStatus(data.metrics.silhouette);
        }

        if (data.points) state.points = data.points;
        if (data.centroids) state.centroids = data.centroids;
        if (data.seed_song) state.seedSong = data.seed_song;
        if (data.recommendations) state.recommendations = data.recommendations;
        if (data.radar) state.radar = data.radar;

        // Update Seed Track Banner
        if (state.seedSong) {
            elements.bannerTitle.textContent = state.seedSong.title;
            elements.bannerArtist.textContent = `${state.seedSong.artist} (${state.seedSong.year})`;
            elements.bannerGenre.textContent = `🏷️ ${state.seedSong.genre}`;
            elements.bannerTempo.textContent = `🥁 ${state.seedSong.tempo} BPM`;

            // Profile Card
            elements.profileSeedTitle.textContent = state.seedSong.title;
            elements.profileSeedArtist.textContent = state.seedSong.artist;
            elements.barSeedDance.style.width = `${state.seedSong.danceability * 100}%`;
            elements.valSeedDance.textContent = state.seedSong.danceability;
            elements.barSeedEnergy.style.width = `${state.seedSong.energy * 100}%`;
            elements.valSeedEnergy.textContent = state.seedSong.energy;
            elements.barSeedValence.style.width = `${state.seedSong.valence * 100}%`;
            elements.valSeedValence.textContent = state.seedSong.valence;
            elements.barSeedAcoustic.style.width = `${state.seedSong.acousticness * 100}%`;
            elements.valSeedAcoustic.textContent = state.seedSong.acousticness;

            // Update dropdown selection
            if (elements.songSelect && state.seedSong.id !== "custom") {
                elements.songSelect.value = state.seedSong.id;
            }

            // Sync quick sample chips
            document.querySelectorAll(".sample-chip").forEach(chip => {
                if (chip.getAttribute("data-song-id") === state.seedSong.id) {
                    chip.classList.add("active");
                } else {
                    chip.classList.remove("active");
                }
            });
        }

        // Update Recommendations Deck (Playlist rows)
        if (state.recommendations) {
            elements.recCountBadge.textContent = `Top ${state.recommendations.length}`;
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
                        <button class="btn-play-seed" data-song-id="${rec.id}" title="Chọn bài này làm Seed">
                            ▶ Chọn
                        </button>
                    </div>
                `;
            });
            elements.recsGrid.innerHTML = recsHtml;

            // Attach listeners to "Chọn" buttons
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
            elements.radarSeedLegend.textContent = state.radar.seed_title || "Seed";
            elements.radarRecLegend.textContent = state.radar.rec_title || "Top Match";
        }

        // Update Latency & Insight
        if (data.metrics && data.metrics.inference_time_ms !== undefined) {
            elements.recLatency.textContent = `Độ trễ: ~${data.metrics.inference_time_ms} ms`;
        }
        if (data.explanation) {
            elements.explanationText.textContent = data.explanation;
        }

        renderLegend();
        renderCanvas();
        renderRadarChart();
    }

    // Trigger Server API for Clustering & Recommendations
    function triggerClusterUpdate() {
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
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                updateUI(data);
            } else {
                console.error("Cluster API error:", data.error);
            }
        })
        .catch(err => console.error("Cluster API network error:", err));
    }

    // Select a New Seed Song
    function selectSeedSong(songId) {
        state.seedId = songId;
        state.customFeatures = null;
        triggerClusterUpdate();
    }

    // Setup All Event Listeners
    function setupEventListeners() {
        // 1. Dropdown Selection
        elements.songSelect.addEventListener("change", (e) => {
            selectSeedSong(e.target.value);
        });

        // 2. Random Seed Song Button
        elements.randomSongBtn.addEventListener("click", () => {
            const options = elements.songSelect.options;
            const randomIndex = Math.floor(Math.random() * options.length);
            elements.songSelect.selectedIndex = randomIndex;
            selectSeedSong(options[randomIndex].value);
        });

        // 3. Curated Quick Sample Chips Ribbon
        elements.samplesScroll.addEventListener("click", (e) => {
            const chip = e.target.closest(".sample-chip");
            if (chip) {
                const songId = chip.getAttribute("data-song-id");
                selectSeedSong(songId);
            }
        });

        // 4. Algorithm Tabs
        elements.algoTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                elements.algoTabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                state.algorithm = tab.getAttribute("data-algo");

                // Toggle visibility
                elements.kmeansControls.classList.toggle("hidden", state.algorithm !== "kmeans");
                elements.hierarchicalControls.classList.toggle("hidden", state.algorithm !== "hierarchical");
                elements.dbscanControls.classList.toggle("hidden", state.algorithm !== "dbscan");

                triggerClusterUpdate();
            });
        });

        // 5. K-Means Sliders
        elements.kSlider.addEventListener("input", (e) => {
            state.k = parseInt(e.target.value);
            elements.kVal.textContent = state.k;
            debounce(triggerClusterUpdate, 200);
        });

        elements.initBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.initBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.init = btn.getAttribute("data-init");
                triggerClusterUpdate();
            });
        });

        // 6. Hierarchical Controls
        elements.hierKSlider.addEventListener("input", (e) => {
            state.k = parseInt(e.target.value);
            elements.hierKVal.textContent = state.k;
            debounce(triggerClusterUpdate, 200);
        });

        elements.linkageSelect.addEventListener("change", (e) => {
            state.linkage = e.target.value;
            triggerClusterUpdate();
        });

        // 7. DBSCAN Controls
        elements.epsSlider.addEventListener("input", (e) => {
            state.eps = parseFloat(e.target.value);
            elements.epsVal.textContent = state.eps.toFixed(2);
            debounce(triggerClusterUpdate, 250);
        });

        elements.minPtsSlider.addEventListener("input", (e) => {
            state.minSamples = parseInt(e.target.value);
            elements.minPtsVal.textContent = state.minSamples;
            debounce(triggerClusterUpdate, 250);
        });

        // 8. Distance Metric & Filter
        elements.metricBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.metricBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.metric = btn.getAttribute("data-metric");
                triggerClusterUpdate();
            });
        });

        elements.filterClusterCheckbox.addEventListener("change", (e) => {
            state.filterCluster = e.target.checked;
            triggerClusterUpdate();
        });

        elements.topNSlider.addEventListener("input", (e) => {
            state.topN = parseInt(e.target.value);
            elements.topNVal.textContent = state.topN;
            debounce(triggerClusterUpdate, 200);
        });

        // 9. Custom Audio Sliders
        const updateSliderDisplay = (slider, display) => {
            slider.addEventListener("input", (e) => {
                display.textContent = e.target.value;
            });
        };
        updateSliderDisplay(elements.sliderDance, elements.sliderDanceVal);
        updateSliderDisplay(elements.sliderEnergy, elements.sliderEnergyVal);
        updateSliderDisplay(elements.sliderValence, elements.sliderValenceVal);
        updateSliderDisplay(elements.sliderTempo, elements.sliderTempoVal);
        updateSliderDisplay(elements.sliderAcoustic, elements.sliderAcousticVal);

        elements.applyCustomBtn.addEventListener("click", () => {
            state.customFeatures = [
                parseFloat(elements.sliderDance.value),
                parseFloat(elements.sliderEnergy.value),
                parseFloat(elements.sliderValence.value),
                parseFloat(elements.sliderTempo.value),
                parseFloat(elements.sliderAcoustic.value),
                0.0,   // instrumentalness
                0.06,  // speechiness
                -6.0   // loudness
            ];
            triggerClusterUpdate();
        });

        // 10. Interactive Canvas Hover & Click
        const canvas = elements.audioSpaceCanvas;
        canvas.addEventListener("mousemove", (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const width = rect.width;
            const height = 480;

            let closestPoint = null;
            let minDist = 14; // hover hit radius

            state.points.forEach(p => {
                const pos = mapCoords(p.x, p.y, width, height);
                const dx = pos.px - mouseX;
                const dy = pos.py - mouseY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minDist) {
                    minDist = dist;
                    closestPoint = p;
                }
            });

            state.hoveredPoint = closestPoint;

            if (closestPoint) {
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
            } else {
                elements.canvasTooltip.style.display = "none";
            }
        });

        canvas.addEventListener("mouseleave", () => {
            state.hoveredPoint = null;
            elements.canvasTooltip.style.display = "none";
        });

        canvas.addEventListener("click", () => {
            if (state.hoveredPoint) {
                selectSeedSong(state.hoveredPoint.id);
            }
        });

        // 11. Modal Triggers
        const openModal = () => {
            elements.curveModal.classList.remove("hidden");
            loadValidationCurves();
        };

        const closeModal = () => {
            elements.curveModal.classList.add("hidden");
        };

        elements.openCurveBtnHeader.addEventListener("click", openModal);
        elements.openCurveBtnSidebar.addEventListener("click", openModal);
        elements.closeCurveBtn.addEventListener("click", closeModal);
        elements.curveModal.addEventListener("click", (e) => {
            if (e.target === elements.curveModal) closeModal();
        });
    }

    // Debounce helper
    function debounce(func, wait) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(func, wait);
    }

    // Load and Render Validation Curves (Elbow & Silhouette)
    function loadValidationCurves() {
        if (state.validationCurves) {
            renderValidationCharts(state.validationCurves);
            return;
        }

        elements.modalBestKText.textContent = "Đang chạy gom cụm K-Means với K từ 2 đến 8...";

        fetch("/api/validation-curve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                state.validationCurves = data;
                renderValidationCharts(data);
            }
        })
        .catch(err => console.error("Validation Curve error:", err));
    }

    function renderValidationCharts(data) {
        elements.modalBestKText.innerHTML = `
            Số cụm tối ưu được đề xuất là <strong>K = ${data.best_k}</strong> 
            (Hệ số Silhouette đạt đỉnh: <strong>${Math.max(...data.silhouette_curve)}</strong>).
        `;

        // 1. Elbow Chart (Chart.js)
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
                    x: { grid: { color: "rgba(34, 48, 74, 0.4)" }, ticks: { color: "#94A3B8" } },
                    y: { grid: { color: "rgba(34, 48, 74, 0.4)" }, ticks: { color: "#94A3B8" } }
                }
            }
        });

        // 2. Silhouette Chart (Chart.js)
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
                    x: { grid: { color: "rgba(34, 48, 74, 0.4)" }, ticks: { color: "#94A3B8" } },
                    y: { grid: { color: "rgba(34, 48, 74, 0.4)" }, ticks: { color: "#94A3B8" } }
                }
            }
        });
    }

    // Start on DOM Ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();

