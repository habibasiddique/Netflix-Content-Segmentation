# Netflix Content Segmentation — Task 4 (Medium)

An unsupervised machine-learning project that groups Netflix titles into content clusters using K-Means.

## Features
- Numerical and categorical feature preparation
- Missing-value imputation, numeric scaling, and one-hot encoding
- K-Means cluster comparison using silhouette scores
- 2D cluster visualization using Truncated SVD
- Cluster summaries and clustered-title CSV output
- Streamlit UI to explore clusters and assign new metadata to a cluster

## Dataset
Download the Netflix Movies and TV Shows dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows), then place `netflix_titles.csv` at `data/netflix_titles.csv`. The dataset is not included in this repository; follow Kaggle's terms.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Generated outputs
- `models/netflix_kmeans_model.joblib`
- `outputs/netflix_clustered_titles.csv`
- `outputs/cluster_summary.csv`
- `outputs/silhouette_scores.csv`
- `outputs/cluster_visualization.png`

## Method
Features: content type, release year, primary genre/category, primary country, duration value, and duration unit. Numeric fields are imputed and standardized; categorical fields are imputed and one-hot encoded. K-Means is tested for k=2 through k=6.

## Important note
Clusters are discovered from the selected metadata and are not official Netflix categories, viewer ratings, or guaranteed genres.
