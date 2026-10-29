from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "netflix_titles.csv"
MODEL_DIR, OUTPUT_DIR = ROOT / "models", ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_data(df):
    needed = ["type", "release_year", "duration", "listed_in", "country"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    X = df[needed].copy()
    X["release_year"] = pd.to_numeric(X["release_year"], errors="coerce")
    duration = X["duration"].fillna("").astype(str).str.strip()
    X["duration_value"] = pd.to_numeric(duration.str.extract(r"(\\d+)")[0], errors="coerce")
    X["duration_unit"] = duration.str.extract(r"\\d+\\s*(.*)$")[0].fillna("Unknown").str.strip()
    X = X.drop(columns=["duration"])
    for col in ["type", "listed_in", "country", "duration_unit"]:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in ["listed_in", "country"]:
        X[col] = X[col].str.split(",").str[0].str.strip().replace("", "Unknown")
    return X

def main():
    print("\\n[1/6] Loading Netflix dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Download netflix_titles.csv from https://www.kaggle.com/datasets/shivamb/netflix-shows and put it in data/.")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print("\\n[2/6] Preparing features...")
    X = prepare_data(df)
    numeric = ["release_year", "duration_value"]
    categorical = ["type", "listed_in", "country", "duration_unit"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical)
    ])
    print("\\n[3/6] Transforming features...")
    matrix = preprocessor.fit_transform(X)
    print(f"Feature matrix shape: {matrix.shape}")
    print("\\n[4/6] Comparing cluster counts using silhouette score...")
    sample_size = min(1500, matrix.shape[0])
    scores = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        score = silhouette_score(matrix, labels, sample_size=sample_size, random_state=42)
        scores.append((k, score))
        print(f"k={k}: silhouette score={score:.4f}")
    best_k = max(scores, key=lambda item: item[1])[0]
    print(f"Selected k={best_k} based on tested silhouette scores.")
    print("\\n[5/6] Training final K-Means model...")
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    results = df.copy()
    results["cluster"] = labels
    results.to_csv(OUTPUT_DIR / "netflix_clustered_titles.csv", index=False)
    pd.DataFrame(scores, columns=["n_clusters", "silhouette_score"]).to_csv(OUTPUT_DIR / "silhouette_scores.csv", index=False)
    summary = results.groupby("cluster").agg(
        title_count=("title", "count"),
        movies=("type", lambda s: (s == "Movie").sum()),
        tv_shows=("type", lambda s: (s == "TV Show").sum()),
        avg_release_year=("release_year", "mean")
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(matrix)
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    plt.figure(figsize=(10, 6))
    for cid in sorted(plot_df.cluster.unique()):
        part = plot_df[plot_df.cluster == cid]
        plt.scatter(part.x, part.y, s=10, alpha=.55, label=f"Cluster {cid}")
    plt.title("Netflix Content Clusters (2D SVD projection)")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_visualization.png", dpi=160); plt.close()
    joblib.dump({"preprocessor": preprocessor, "model": model}, MODEL_DIR / "netflix_kmeans_model.joblib")
    print("\\n[6/6] Evaluation and saving complete.")
    print(summary.to_string(index=False))
    print(f"Saved model: {MODEL_DIR / 'netflix_kmeans_model.joblib'}")
    print(f"Saved outputs in: {OUTPUT_DIR}")
    print("DONE! Clusters are exploratory groups, not official Netflix categories.")

if __name__ == "__main__":
    main()
