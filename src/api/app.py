import logging
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [LOGS_DIR, MODEL_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommendation API",
    description="Async API for Collaborative Filtering and Content-Based Recommendations",
    version="1.0"
)

try:
    logger.info("Loading CF embeddings and mappings...")
    user_embeddings = joblib.load(MODEL_DIR / "user_embeddings.pkl")
    movie_embeddings = joblib.load(MODEL_DIR / "movie_embeddings.pkl")
    user2idx = joblib.load(MODEL_DIR / "user2idx.pkl")
    movie2idx = joblib.load(MODEL_DIR / "movie2idx.pkl")
except Exception as e:
    logger.warning(f"CF model not fully loaded: {e}")
    user_embeddings = movie_embeddings = user2idx = movie2idx = None

try:
    logger.info("Loading TF-IDF vectorizer and similarity matrix...")
    tfidf_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(MODEL_DIR / "tfidf_similarity_matrix.pkl")
except Exception as e:
    logger.warning(f"CB model not loaded: {e}")
    tfidf_vectorizer = tfidf_matrix = None

movies_path = DATA_DIR / "movies.csv"
if movies_path.exists():
    movies_df = pd.read_csv(movies_path)
else:
    logger.warning(f"Movies metadata not found at {movies_path}")
    movies_df = pd.DataFrame(columns=["movieId", "title", "genres"])


@app.get("/recommend/{user_id}")
async def recommend_cf(user_id: int):
    if any(x is None for x in [user_embeddings, movie_embeddings, user2idx, movie2idx]):
        return {"error": "Collaborative filtering model not loaded properly."}

    if user_id not in user2idx:
        return {"error": f"User ID {user_id} not found."}

    user_idx = user2idx[user_id]
    user_vector = user_embeddings[user_idx]

    # Compute cosine similarity between user vector and all movie vectors
    similarities = np.dot(movie_embeddings, user_vector)
    top_indices = np.argsort(similarities)[-10:][::-1]

    movie_ids = [list(movie2idx.keys())[i] for i in top_indices]
    recommended = movies_df[movies_df["movieId"].isin(movie_ids)][["movieId", "title"]]

    return {"user_id": user_id, "recommendations": recommended.to_dict(orient="records")}


@app.get("/recommend/content/{movie_id}")
async def recommend_content(movie_id: int):
    if tfidf_matrix is None or tfidf_vectorizer is None:
        return {"error": "Content-based model not loaded."}

    if movie_id not in movies_df["movieId"].values:
        return {"error": f"Movie ID {movie_id} not found."}

    idx = movies_df.index[movies_df["movieId"] == movie_id][0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-11:-1][::-1]

    recommended = movies_df.iloc[similar_indices][["movieId", "title"]]
    return {"movie_id": movie_id, "recommendations": recommended.to_dict(orient="records")}
