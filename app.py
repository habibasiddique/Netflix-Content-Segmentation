from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "netflix_kmeans_model.joblib"
SUMMARY_PATH = ROOT / "outputs" / "cluster_summary.csv"
CLUSTERED_PATH = ROOT / "outputs" / "netflix_clustered_titles.csv"
PLOT_PATH = ROOT / "outputs" / "cluster_visualization.png"

st.set_page_config(page_title="Netflix Content Segmentation", page_icon="🎬", layout="wide")
st.title("🎬 Netflix Content Segmentation")
st.caption("Unsupervised machine learning with K-Means clustering")
if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python main.py` in the project terminal first.")
    st.stop()
bundle = joblib.load(MODEL_PATH)
preprocessor, model = bundle["preprocessor"], bundle["model"]
tab1, tab2, tab3 = st.tabs(["Explore clusters", "Assign a title", "About"])
with tab1:
    st.subheader("Cluster overview")
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("cluster")[["movies", "tv_shows"]])
    if PLOT_PATH.exists():
        st.image(str(PLOT_PATH), caption="2D projection of content features, colored by cluster")
    if CLUSTERED_PATH.exists():
        data = pd.read_csv(CLUSTERED_PATH)
        clusters = sorted(data["cluster"].dropna().unique().tolist())
        selected = st.selectbox("Choose a cluster", clusters)
        cols = [c for c in ["title", "type", "release_year", "rating", "duration", "listed_in", "country"] if c in data.columns]
        st.dataframe(data.loc[data.cluster == selected, cols].head(100), use_container_width=True, hide_index=True)
with tab2:
    st.subheader("Assign metadata to a cluster")
    with st.form("prediction_form"):
        content_type = st.selectbox("Content type", ["Movie", "TV Show"])
        release_year = st.number_input("Release year", min_value=1920, max_value=2035, value=2020)
        duration_value = st.number_input("Duration value", min_value=1, max_value=1000, value=100, help="Minutes for movies, seasons for TV shows.")
        duration_unit = st.selectbox("Duration unit", ["min", "Season", "Seasons"])
        genre = st.text_input("Primary genre/category", "Dramas")
        country = st.text_input("Primary country", "United States")
        submitted = st.form_submit_button("Find cluster")
    if submitted:
        row = pd.DataFrame([{"type": content_type, "release_year": release_year, "listed_in": genre, "country": country, "duration_value": duration_value, "duration_unit": duration_unit}])
        cluster = int(model.predict(preprocessor.transform(row))[0])
        st.success(f"This title is assigned to Cluster {cluster}.")
        st.info("Clusters are learned from dataset metadata; they are not official Netflix labels.")
with tab3:
    st.markdown("""
    **Goal:** Group Netflix titles into meaningful clusters using unsupervised machine learning.

    **Workflow**
    1. Prepare numerical and categorical metadata.
    2. Impute missing values, scale numeric fields, and one-hot encode categories.
    3. Compare K-Means cluster counts using silhouette scores.
    4. Visualize clusters in a 2D Truncated SVD projection.
    5. Explore cluster sizes and sample titles.

    **Interpretation:** Clusters depend on chosen features and preprocessing. They are exploratory similarity groups, not ground-truth labels.
    """)
