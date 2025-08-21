# Vehicle-Classification-using-Unsupervised-Machine-Learning

# 🚗 Project Title

Vehicle Type Clustering Using Unsupervised Machine Learning

⸻

# 🎯 Goal

To explore whether natural groupings of vehicles (bus, car, van) can be discovered without labels, using clustering techniques and dimensionality reduction methods.

⸻

# 💼 Business Context

In many real-world applications, labeled data is scarce or expensive to obtain. For example, traffic surveillance systems may capture thousands of vehicle images/features daily, but manual labeling into “bus”, “car”, or “van” is costly. An unsupervised learning approach could help:

	•	Automatically cluster vehicles into groups,
	•	Provide insights into data structure,
	•	Serve as a preprocessing step before supervised learning, or
	•	Detect novel/unseen vehicle types.

⸻

# 🌍 Real-World Impact

	•	Traffic Monitoring: Enables city planners to analyze vehicle flow patterns even when ground-truth labels are missing.
	•	Fleet Management: Helps group vehicles by type for usage analysis without prior classification.
	•	Autonomous Systems: Provides unsupervised pre-training signals when labeled datasets are limited.
	•	Data Annotation Efficiency: Clusters can be used to pre-label data, reducing manual annotation costs.

# 📌 Project Insight

This project explores the use of unsupervised learning (clustering) on the Vehicle dataset to identify natural groupings among vehicles (Bus, Car, Van) without labels.
The aim is to see whether clustering methods like KMeans + PCA can reveal meaningful separation of vehicle classes and provide business insights when labeled data is unavailable.

⸻

# 🎯 Workflow

## 1️⃣ Importing Libraries

All required libraries for EDA, preprocessing, dimensionality reduction, clustering, and evaluation are imported (NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, etc.).

⸻

## 2️⃣ Loading the Dataset

	•	Dataset used: Vehicle Dataset (multiclass: Bus, Car, Van).
	•	Initial shape: (846, 19) (before preprocessing).
**dataset link:** https://drive.google.com/file/d/11kXoTwaCsDHxCYSA-Cs8JugxY_kNUqjQ/view?usp=sharing

⸻

## 3️⃣ Exploratory Data Analysis (EDA)
	•	Basic info: data types, shape, duplicates, nulls.
	•	Descriptive statistics: mean, std, min, max.
	•	Target balance: class distribution across Bus, Car, Van.
	•	Univariate distributions: histograms/boxplots for key features.
	•	Feature separation summary:
	•	Cars have higher circularity.
	•	Vans show wider radius ratio.
	•	Buses show distinct scatter ratio.

⸻

## 4️⃣ Preprocessing
	•	Handling Null Values → no major missing values found.
	•	Capping Outliers → extreme outliers capped using IQR.
	•	Noise Detection (Isolation Forest) → reduced dataset from 846 → 803 samples by removing noisy points.
	•	Encoding/Feature Transformation → categorical → numeric encoding.
	•	Feature Scaling → standardized features using StandardScaler.

⸻

## 5️⃣ Unsupervised EDA
	•	PCA (Principal Component Analysis) → retained ~80% variance with top components.
	•	t-SNE & UMAP → for visualization of non-linear separability.
	•	Pairplots of Principal Components → checked natural grouping.

⸻

## 6️⃣ Clustering (KMeans)
	•	Hopkins statistic = 1.0 → indicates clustering tendency.
	•	Elbow Method + Silhouette + Davies–Bouldin → suggested k=2 or k=3 as reasonable choices.
	•	KMeans applied on PCA-reduced dataset.
	•	Pairplots with cluster coloring (PC1–PC3) → partial separation observed.

⸻

## 7️⃣ Evaluation

Metrics Used:

	•	Adjusted Rand Index (ARI): 0.1219 → very poor alignment with actual labels.
	•	Silhouette Score (Raw KMeans): 0.3015 → weak clustering (overlaps).
	•	Silhouette Score (KMeans + PCA): 0.4124 → moderate cluster separation.
	•	Weighted F1 Score: 0.24 → very poor classification performance.

## 📊 Class-wise Feature Mean Summary Table

	•	Bus → higher scatter ratio, radius ratio.
	•	Car → high circularity, variance.
	•	Van → intermediate across most features.

⸻

## 8️⃣ Conclusion
	•	Unsupervised learning was not effective for precise classification of Bus, Car, and Van.
	•	Clusters showed moderate separability (PCA + KMeans) but poor alignment with ground-truth labels.
	•	Best use of this pipeline:
	•	Data exploration when labels are unavailable.
	•	Pre-labeling to speed up manual annotation.
	•	Dimensionality reduction & noise detection as preprocessing steps before supervised ML.

# ⚠️ Comparison with Supervised Models:

Supervised ML models (like XGBoost, SVC, Random Forest) achieved >93% accuracy, proving far superior.
Unsupervised models remain valuable for structural insight and data preparation, but not final classification.
