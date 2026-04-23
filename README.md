# Food-Delivery-Time-Prediction
Predicting food delivery times using Machine Learning. This repository showcases end-to-end data pipelines, in-depth EDA, and smart feature engineering (e.g., matching courier skill to route difficulty) to build robust, real-world estimation models 

  <div align="center">

<img src="https://www.google.com/search?q=https://images.unsplash.com/photo-1558981403-c5f9899a28bc%3Fauto%3Dformat%26fit%3Dcrop%26q%3D80%26w%3D1000" alt="AI Delivery Rider" width="100%" style="border-radius: 12px; max-height: 400px; object-fit: cover; border: 2px solid #06b6d4; box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);">

<h1 style="color: #06b6d4; font-family: monospace;">>_ PREDICTIVE ROUTE INTELLIGENCE</h1>
<p><b>A Neural-Enhanced Machine Learning Pipeline for Delivery Logistics</b></p>

</div>

🌌 SYSTEM OVERVIEW

Late deliveries compromise system efficiency and user retention. This project analyzes 1,000 historical delivery nodes to classify delivery states as Delayed (1) or On-Time (0) applying a strict 60-minute temporal threshold.

The pipeline evaluates highly dimensional environmental, route, and courier vectors to achieve a top confidence score of 95.4% ROC AUC.

📊 Target State Distribution

(Dynamic Mermaid Chart)

pie title Delivery Status Imbalance (60m Threshold)
  "On-Time (0)" : 58.5
  "Delayed (1)" : 41.5


🔬 PREPROCESSING & SYNTHESIS

To maximize predictive capabilities, the raw data underwent rigorous noise reduction and dimensional scaling:

Noise Reduction: Localized and repaired 3% data corruption via Modal/Median imputation. Outliers capped via 99th percentile IQR.

Dimensional Scaling: Standardized spatial metrics via Z-score, bounded courier experience via Min-Max, and applied Robust scaling to highly skewed axes.

Feature Engineering:

Dist/Exp Ratio: Route difficulty matrix.

Load Score: Weighted complexity algorithm.

Env. Stress: Multi-variable weather/traffic interaction.

🤖 ALGORITHMIC EVALUATION

We trained 7 discrete classification topologies via three rigorous validation protocols (70:30, 60:20:20, and 5-Fold Stratified Cross Validation) to ensure robust generalization across unseen environment states.

📈 Performance Matrix (5-Fold CV Mean AUC)

Visualizing algorithm accuracy metrics:

Neural Architecture

Confidence Level

Visual Weight

Neural Network

0.954

🟪🟪🟪🟪🟪🟪🟪🟪🟪🟪 95.4%

Logistic Regression

0.953

🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 95.3%

Random Forest

0.952

🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 95.2%

SVM

0.950

🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩 95.0%

Gradient Boosting

0.945

🟨🟨🟨🟨🟨🟨🟨🟨🟨⬛ 94.5%

KNN

0.926

🟧🟧🟧🟧🟧🟧🟧🟧⬛⬛ 92.6%

Decision Tree

0.917

🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛ 91.7%

🏆 Optimal Architectures

Logistic Regression > Why: Supreme structural consistency. Highly interpretable parameters ideal for operational deployment and extremely rapid real-time inference without hyperparameter tuning.

Support Vector Machine (SVM)

Why: Achieved peak test set accuracy. The RBF kernel maps perfectly with normalized dimensional spaces, offering maximum resilience to data overfitting.

⚖️ FEATURE WEIGHT ANALYSIS

Random Forest extraction revealed that our custom-engineered vectors generate ~45% of the total predictive logic.

xychart-beta
    title "Calculated Decision Weight by Feature"
    x-axis ["Distance", "Load Score", "Dist/Exp Ratio", "Prep Time", "Env. Stress", "Courier Exp."]
    y-axis "Importance Score" 0.00 --> 0.35
    bar [0.31, 0.29, 0.15, 0.06, 0.03, 0.03]


⚡ SYSTEM DIRECTIVES

Based on algorithmic weight analysis, we recommend the following operational interventions for the routing matrix:

⏱️ Dynamic Time Offsets: Apply algorithmic padding of +15-20% for radial distances >15km. Activate dynamic environmental buffers during anomalous weather/traffic states.

👨‍🚀 Node Assignment Logic: Map high-tier couriers (5+ yrs) to high-complexity vectors (>12km) leveraging the custom Distance_per_Experience parameter.

🚨 Automated Intervention: Initialize the Logistic Regression model in the production environment to trigger proactive user communication APIs when the delay probability surpasses a 0.70 threshold.

<div align="center">
<code>Predictive Algorithm Documentation &copy; AI Frameworks</code>
</div>
