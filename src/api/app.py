import logging
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List, Dict, Any
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_DIR = BASE_DIR / "data"
MOVIES_CANDIDATES = [DATA_DIR / "processed" / "movies.csv", DATA_DIR / "movies.csv"]
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
FRONTEND_DIR = BASE_DIR / "src" / "frontend"

for d in (LOGS_DIR, MODEL_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=LOGS_DIR / "api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("movie_recs_api")
logger.addHandler(logging.StreamHandler())  # also log to console

app = FastAPI(
    title="Movie Recommendation API",
    description="Async API for Collaborative Filtering (embeddings) and Content-Based (TF-IDF) recommendations",
    version="1.0",
)

# app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# @app.get("/", response_class=FileResponse)
# async def serve_frontend():
#     return FRONTEND_DIR / "index.html"

def safe_joblib_load(path: Path):
    try:
        obj = joblib.load(path)
        logger.info(f"Loaded: {path}")
        return obj
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None

movies_path = None
for p in MOVIES_CANDIDATES:
    if p.exists():
        movies_path = p
        break

if movies_path is None:
    logger.warning(f"No movies.csv found in candidates: {MOVIES_CANDIDATES}")
    movies_df = pd.DataFrame(columns=["movieId", "title", "genres"])
else:
    movies_df = pd.read_csv(movies_path)
    movies_df["title_norm"] = movies_df["title"].astype(str).str.strip().str.lower()
    logger.info(f"Loaded movies metadata: {movies_path} ({len(movies_df)} rows)")

user_emb_dict = safe_joblib_load(MODEL_DIR / "user_embeddings.pkl")
movie_emb_dict = safe_joblib_load(MODEL_DIR / "movie_embeddings.pkl")
user2idx = safe_joblib_load(MODEL_DIR / "user2idx.pkl")
movie2idx = safe_joblib_load(MODEL_DIR / "movie2idx.pkl")

user_embeddings = None
movie_embeddings = None
idx2user = {}
idx2movie = {}

if user_emb_dict is not None and user2idx is not None:
    max_user_idx = max(user2idx.values())
    emb_dim = None
    for uid, vec in user_emb_dict.items():
        emb_dim = len(vec)
        break
    if emb_dim is None:
        logger.warning("User embeddings dict empty.")
    else:
        user_embeddings = np.zeros((max_user_idx + 1, emb_dim), dtype=float)
        for raw_uid, idx in user2idx.items():
            vec = user_emb_dict.get(raw_uid)
            if vec is None:
                vec = user_emb_dict.get(str(raw_uid))
            if vec is None:
                logger.debug(f"No vector found for user id {raw_uid}; leaving zeros")
                continue
            user_embeddings[idx] = np.asarray(vec, dtype=float)
            idx2user[idx] = raw_uid
        logger.info(f"user_embeddings matrix shape: {user_embeddings.shape}")

if movie_emb_dict is not None and movie2idx is not None:
    max_movie_idx = max(movie2idx.values())
    emb_dim = None
    for mid, vec in movie_emb_dict.items():
        emb_dim = len(vec)
        break
    if emb_dim is None:
        logger.warning("Movie embeddings dict empty.")
    else:
        movie_embeddings = np.zeros((max_movie_idx + 1, emb_dim), dtype=float)
        for raw_mid, idx in movie2idx.items():
            vec = movie_emb_dict.get(raw_mid)
            if vec is None:
                vec = movie_emb_dict.get(str(raw_mid))
            if vec is None:
                logger.debug(f"No vector for movie id {raw_mid}; leaving zeros")
                continue
            movie_embeddings[idx] = np.asarray(vec, dtype=float)
            idx2movie[idx] = raw_mid
        logger.info(f"movie_embeddings matrix shape: {movie_embeddings.shape}")


tfidf_vectorizer = safe_joblib_load(MODEL_DIR / "tfidf_vectorizer.pkl")
tfidf_matrix = safe_joblib_load(MODEL_DIR / "tfidf_similarity_matrix.pkl")  # your name

if tfidf_matrix is not None:
    tfidf_matrix = np.asarray(tfidf_matrix)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

def find_movie_by_title(query_title: str, cutoff: float = 0.6) -> Optional[str]:
    """Return the exact title found in movies_df (normalized) or the closest match."""
    q = query_title.strip().lower()
    if q in set(movies_df["title_norm"].values):
        return movies_df.loc[movies_df["title_norm"] == q, "title"].iloc[0]
  
    choices = movies_df["title_norm"].tolist()
    matches = get_close_matches(q, choices, n=1, cutoff=cutoff)
    if matches:
        return movies_df.loc[movies_df["title_norm"] == matches[0], "title"].iloc[0]
    return None


@app.get("/")
async def root():
    return {
        "message": "Movie Recommendation API",
        "models": {
            "cf_loaded": user_embeddings is not None and movie_embeddings is not None,
            "cb_loaded": tfidf_matrix is not None and tfidf_vectorizer is not None,
            "movies_rows": len(movies_df),
        },
    }

#Collaborative Filtering endpoint 
@app.get("/recommend/{user_id}")
async def recommend_cf(user_id: int, top_n: int = 10):

    user_idx = user2idx.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail="User ID not found")

    user_vec = user_embeddings[user_idx]

    scores = np.dot(movie_embeddings, user_vec)

    top_indices = np.argsort(scores)[-top_n:][::-1]

    recommended_movies = []
    for idx in top_indices:
        movie_id = int(list(movie2idx.keys())[list(movie2idx.values()).index(idx)])
        movie_title = str(movies_df.loc[movies_df["movieId"] == movie_id]["title"].values[0])

        recommended_movies.append({
            "movie_id": movie_id,          # convert numpy → int
            "title": movie_title,
            "score": float(scores[idx])    # convert numpy → float
        })

    return {"user_id": int(user_id), "recommendations": recommended_movies}


# Content-Based endpoint 
@app.get("/recommend/content/")
async def recommend_content(
    movie_id: Optional[int] = Query(None, description="movieId (int)"),
    title: Optional[str] = Query(None, description="movie title (string)"),
    top_n: int = Query(10, ge=1, le=100),
):
    """
    Recommend top_n movies similar to given movie_id or title using TF-IDF cosine similarity.
    Provide either movie_id OR title (title allows fuzzy matching).
    """
    if tfidf_matrix is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=500, detail="Content-based artifacts not loaded.")

    if movie_id is None and title is None:
        raise HTTPException(status_code=400, detail="Provide either movie_id or title.")

    if movie_id is not None:
        if movie_id not in movies_df["movieId"].values:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found.")
        idx = int(movies_df.index[movies_df["movieId"] == movie_id][0])
    else:
        found_title = find_movie_by_title(title)
        if found_title is None:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found (no close match).")
        idx = int(movies_df.index[movies_df["title"] == found_title][0])

    sims = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    top_idx = sims.argsort()[-(top_n + 1):][::-1]  
    top_idx = [i for i in top_idx if i != idx][:top_n]

    recs = []
    for i in top_idx:
        recs.append({
            "movieId": int(movies_df.iloc[i]["movieId"]),
            "title": str(movies_df.iloc[i]["title"]),
            "score": float(sims[i]),
        })

    return {"query_index": int(idx), "recommendations": recs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
