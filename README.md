# Food-Delivery-Time-Prediction
Predicting food delivery times using Machine Learning. This repository showcases end-to-end data pipelines, in-depth EDA, and smart feature engineering (e.g., matching courier skill to route difficulty) to build robust, real-world estimation models 

  <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Route Intelligence - AI Report</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Chart.js for Live Interactive Graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #020617; /* Slate 950 */
            color: #cbd5e1; /* Slate 300 */
        }
        h1, h2, h3, h4, .font-mono {
            font-family: 'Fira Code', monospace;
        }
        .glow-border {
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
        }
        .markdown-container h2 {
            color: #22d3ee; /* Cyan 400 */
            font-size: 1.75rem;
            font-weight: 700;
            margin-top: 3rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(34, 211, 238, 0.2);
            padding-bottom: 0.5rem;
        }
        .markdown-container h3 {
            color: #e2e8f0;
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .markdown-container p {
            margin-bottom: 1.25rem;
            line-height: 1.75;
        }
        .markdown-container ul {
            list-style-type: disc;
            padding-left: 1.5rem;
            margin-bottom: 1.25rem;
            line-height: 1.75;
        }
        .markdown-container li {
            margin-bottom: 0.5rem;
        }
        .markdown-container strong {
            color: #f8fafc;
        }
        .markdown-container code {
            background-color: #0f172a;
            color: #c084fc;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.875em;
            border: 1px solid #1e293b;
        }
        .markdown-container blockquote {
            border-left: 4px solid #22d3ee;
            padding-left: 1rem;
            color: #94a3b8;
            background-color: rgba(15, 23, 42, 0.5);
            padding: 1rem;
            border-radius: 0 0.5rem 0.5rem 0;
            margin-bottom: 1.25rem;
        }
        .chart-container {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin: 2rem 0;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }
    </style>
</head>
<body class="antialiased selection:bg-cyan-900 selection:text-cyan-100 p-6 md:p-12 lg:p-16">

    <main class="max-w-4xl mx-auto markdown-container">
        
        <!-- Header Section -->
        <div class="text-center mb-12">
            <div class="relative inline-block w-full mb-8 group">
                <div class="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
                <img src="https://images.unsplash.com/photo-1558981403-c5f9899a28bc?auto=format&fit=crop&q=80&w=1200" 
                     alt="AI Delivery Rider" 
                     class="relative rounded-xl border border-slate-700 w-full h-[300px] md:h-[450px] object-cover glow-border mix-blend-luminosity hover:mix-blend-normal transition-all duration-700">
            </div>
            
            <h1 class="text-3xl md:text-5xl text-cyan-400 font-bold mb-4 tracking-tight">>_ PREDICTIVE ROUTE INTELLIGENCE</h1>
            <p class="text-lg md:text-xl text-slate-400"><b>A Neural-Enhanced Machine Learning Pipeline for Delivery Logistics</b></p>
            
            <div class="flex flex-wrap justify-center gap-3 mt-6">
                <span class="bg-slate-900 border border-slate-700 text-cyan-400 px-3 py-1 rounded text-sm font-mono flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span> Data Vectors: 1,000</span>
                <span class="bg-slate-900 border border-slate-700 text-purple-400 px-3 py-1 rounded text-sm font-mono flex items-center gap-2">Features: 9</span>
                <span class="bg-slate-900 border border-slate-700 text-emerald-400 px-3 py-1 rounded text-sm font-mono flex items-center gap-2">Top Model: SVM / LogReg</span>
            </div>
        </div>

        <!-- Content Sections -->
        <h2>🌌 SYSTEM OVERVIEW</h2>
        <p>Late deliveries compromise system efficiency and user retention. This project analyzes <strong>1,000 historical delivery nodes</strong> to classify delivery states as <strong class="text-rose-400">Delayed (1)</strong> or <strong class="text-emerald-400">On-Time (0)</strong> applying a strict 60-minute temporal threshold.</p>
        <p>The pipeline evaluates highly dimensional environmental, route, and courier vectors to achieve a top confidence score of <strong>95.4% ROC AUC</strong>.</p>

        <!-- LIVE CHART 1 -->
        <h3>📊 Target State Distribution</h3>
        <p class="text-sm text-slate-500 italic">Hover over the segments to view live percentages.</p>
        <div class="chart-container">
            <div class="relative w-full h-[300px] flex justify-center">
                <canvas id="targetChart"></canvas>
            </div>
        </div>

        <h2>🔬 PREPROCESSING & SYNTHESIS</h2>
        <p>To maximize predictive capabilities, the raw data underwent rigorous noise reduction and dimensional scaling:</p>
        <ul>
            <li><strong>Noise Reduction:</strong> Localized and repaired 3% data corruption via Modal/Median imputation. Outliers capped via 99th percentile IQR.</li>
            <li><strong>Dimensional Scaling:</strong> Standardized spatial metrics via Z-score, bounded courier experience via Min-Max, and applied Robust scaling to highly skewed axes.</li>
            <li><strong>Feature Engineering:</strong>
                <ul>
                    <li><code>Dist/Exp Ratio:</code> Route difficulty matrix.</li>
                    <li><code>Load Score:</code> Weighted complexity algorithm.</li>
                    <li><code>Env. Stress:</code> Multi-variable weather/traffic interaction.</li>
                </ul>
            </li>
        </ul>

        <h2>🤖 ALGORITHMIC EVALUATION</h2>
        <p>We trained 7 discrete classification topologies via three rigorous validation protocols (70:30, 60:20:20, and 5-Fold Stratified Cross Validation) to ensure robust generalization across unseen environment states.</p>

        <!-- LIVE CHART 2 -->
        <h3>📈 Performance Matrix (5-Fold CV Mean AUC)</h3>
        <p class="text-sm text-slate-500 italic">Interactive bar chart comparing structural confidence levels. Hover for exact metrics.</p>
        <div class="chart-container">
            <div class="relative w-full h-[350px]">
                <canvas id="modelChart"></canvas>
            </div>
        </div>

        <h3>🏆 Optimal Architectures</h3>
        <p><strong>1. Logistic Regression</strong></p>
        <blockquote>
            <strong>Why:</strong> Supreme structural consistency. Highly interpretable parameters ideal for operational deployment and extremely rapid real-time inference without hyperparameter tuning.
        </blockquote>
        <p><strong>2. Support Vector Machine (SVM)</strong></p>
        <blockquote>
            <strong>Why:</strong> Achieved peak test set accuracy. The RBF kernel maps perfectly with normalized dimensional spaces, offering maximum resilience to data overfitting.
        </blockquote>

        <h2>⚖️ FEATURE WEIGHT ANALYSIS</h2>
        <p>Random Forest extraction revealed that our custom-engineered vectors generate <strong>~45% of the total predictive logic</strong>.</p>

        <!-- LIVE CHART 3 -->
        <p class="text-sm text-slate-500 italic">Live horizontal chart calculating decision weight by feature variable.</p>
        <div class="chart-container">
            <div class="relative w-full h-[350px]">
                <canvas id="featureChart"></canvas>
            </div>
        </div>

        <h2>⚡ SYSTEM DIRECTIVES</h2>
        <p>Based on algorithmic weight analysis, we recommend the following operational interventions for the routing matrix:</p>
        <ol class="list-decimal pl-5 space-y-4 mb-12">
            <li><strong class="text-cyan-300">Dynamic Time Offsets:</strong> Apply algorithmic padding of <code>+15-20%</code> for radial distances <code>>15km</code>. Activate dynamic environmental buffers during anomalous weather/traffic states.</li>
            <li><strong class="text-purple-300">Node Assignment Logic:</strong> Map high-tier couriers (5+ yrs) to high-complexity vectors (<code>>12km</code>) leveraging the custom <code>Distance_per_Experience</code> parameter.</li>
            <li><strong class="text-emerald-300">Automated Intervention:</strong> Initialize the <strong>Logistic Regression</strong> model in the production environment to trigger proactive user communication APIs when the delay probability surpasses a <code>0.70</code> threshold.</li>
        </ol>

        <div class="text-center py-8 border-t border-slate-800 text-sm text-slate-500 font-mono">
            <p>Predictive Algorithm Documentation &copy; AI Frameworks 2026</p>
        </div>
    </main>

    <!-- Chart.js Logic for Live Graphs -->
    <script>
        // Set Global Styling for Charts to match Dark/AI Theme
        Chart.defaults.font.family = "'Fira Code', 'Inter', monospace";
        Chart.defaults.color = '#94a3b8'; // Slate 400

        // 1. Doughnut Chart (Target Distribution)
        const ctxTarget = document.getElementById('targetChart').getContext('2d');
        new Chart(ctxTarget, {
            type: 'doughnut',
            data: {
                labels: ['On-Time (<60m)', 'Delayed (>60m)'],
                datasets: [{
                    data: [58.5, 41.5],
                    backgroundColor: ['rgba(16, 185, 129, 0.8)', 'rgba(244, 63, 94, 0.8)'], // Emerald & Rose
                    borderColor: '#0f172a',
                    borderWidth: 3,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 20, usePointStyle: true, color: '#e2e8f0' }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#22d3ee',
                        bodyColor: '#f8fafc',
                        borderColor: 'rgba(34, 211, 238, 0.3)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) { return ` Status: ${context.raw}% of Nodes`; }
                        }
                    }
                },
                animation: { animateScale: true, animateRotate: true, duration: 1500 }
            }
        });

        // 2. Bar Chart (Model Performance Matrix)
        const ctxModel = document.getElementById('modelChart').getContext('2d');
        const gradCyan = ctxModel.createLinearGradient(0, 0, 0, 400);
        gradCyan.addColorStop(0, '#22d3ee'); gradCyan.addColorStop(1, '#0284c7');
        const gradPurple = ctxModel.createLinearGradient(0, 0, 0, 400);
        gradPurple.addColorStop(0, '#c084fc'); gradPurple.addColorStop(1, '#7e22ce');

        new Chart(ctxModel, {
            type: 'bar',
            data: {
                labels: ['Neural Net', 'Logistic Reg.', 'Random Forest', 'SVM', 'Grad Boost', 'KNN', 'Decision Tree'],
                datasets: [{
                    label: 'ROC AUC Score',
                    data: [0.954, 0.953, 0.952, 0.950, 0.945, 0.926, 0.917],
                    backgroundColor: [gradPurple, gradCyan, gradCyan, gradCyan, '#475569', '#334155', '#1e293b'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        min: 0.85, max: 1.0,
                        grid: { color: 'rgba(255,255,255,0.05)', borderDash: [5, 5] },
                        ticks: { font: { weight: '600' } }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        borderColor: 'rgba(192, 132, 252, 0.3)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) { return ` Confidence Level: ${(context.raw * 100).toFixed(1)}%`; }
                        }
                    }
                },
                animation: { duration: 2000, easing: 'easeOutBounce' }
            }
        });

        // 3. Horizontal Bar Chart (Feature Importance)
        const ctxFeature = document.getElementById('featureChart').getContext('2d');
        const gradBlueH = ctxFeature.createLinearGradient(0, 0, 400, 0);
        gradBlueH.addColorStop(0, '#0ea5e9'); gradBlueH.addColorStop(1, '#38bdf8');

        new Chart(ctxFeature, {
            type: 'bar',
            data: {
                labels: ['Distance (km)', 'Total Load Score', 'Dist/Exp Ratio', 'Prep Time', 'Env. Stress', 'Courier Exp.'],
                datasets: [{
                    label: 'Decision Weight',
                    data: [0.3122, 0.2964, 0.1501, 0.0600, 0.0384, 0.0350],
                    backgroundColor: [gradBlueH, gradPurple, gradPurple, '#475569', gradPurple, '#334155'],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y', // Makes the chart horizontal
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { 
                        max: 0.35,
                        grid: { color: 'rgba(255,255,255,0.05)', borderDash: [5, 5] },
                        title: { display: true, text: 'Importance Metric', color: '#cbd5e1' }
                    },
                    y: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        borderColor: 'rgba(56, 189, 248, 0.3)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) { return ` Weight: ${(context.raw * 100).toFixed(1)}%`; }
                        }
                    }
                },
                animation: { duration: 1500, delay: 500 }
            }
        });
    </script>
</body>
</html>
👨‍🚀 Node Assignment Logic: Map high-tier couriers (5+ yrs) to high-complexity vectors (>12km) leveraging the custom Distance_per_Experience parameter.

🚨 Automated Intervention: Initialize the Logistic Regression model in the production environment to trigger proactive user communication APIs when the delay probability surpasses a 0.70 threshold.

<div align="center">
<code>Predictive Algorithm Documentation &copy; AI Frameworks</code>
</div>
