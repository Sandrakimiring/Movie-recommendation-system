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
    description="Async API for Collaborative Filtering, Content-Based, and Hybrid recommendations",
    version="1.1",
)

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

# Load Models
user_emb_dict = safe_joblib_load(MODEL_DIR / "user_embeddings.pkl")
movie_emb_dict = safe_joblib_load(MODEL_DIR / "movie_embeddings.pkl")
user2idx = safe_joblib_load(MODEL_DIR / "user2idx.pkl")
movie2idx = safe_joblib_load(MODEL_DIR / "movie2idx.pkl")

tfidf_vectorizer = safe_joblib_load(MODEL_DIR / "tfidf_vectorizer.pkl")
tfidf_matrix = safe_joblib_load(MODEL_DIR / "tfidf_matrix.pkl") # Needed for hybrid if we want to compute on fly, but we use profiles
tfidf_sim_matrix = safe_joblib_load(MODEL_DIR / "tfidf_similarity_matrix.pkl")
user_content_profiles = safe_joblib_load(MODEL_DIR / "user_content_profiles.pkl")

# Prepare CF Matrices
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
    if emb_dim:
        user_embeddings = np.zeros((max_user_idx + 1, emb_dim), dtype=float)
        for raw_uid, idx in user2idx.items():
            vec = user_emb_dict.get(raw_uid)
            if vec is None:
                vec = user_emb_dict.get(str(raw_uid))
            if vec is not None:
                user_embeddings[idx] = np.asarray(vec, dtype=float)
                idx2user[idx] = raw_uid

if movie_emb_dict is not None and movie2idx is not None:
    max_movie_idx = max(movie2idx.values())
    emb_dim = None
    for mid, vec in movie_emb_dict.items():
        emb_dim = len(vec)
        break
    if emb_dim:
        movie_embeddings = np.zeros((max_movie_idx + 1, emb_dim), dtype=float)
        for raw_mid, idx in movie2idx.items():
            vec = movie_emb_dict.get(raw_mid)
            if vec is None:
                vec = movie_emb_dict.get(str(raw_mid))
            if vec is not None:
                movie_embeddings[idx] = np.asarray(vec, dtype=float)
                idx2movie[idx] = raw_mid

def find_movie_by_title(query_title: str, cutoff: float = 0.6) -> Optional[str]:
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
            "cb_loaded": tfidf_sim_matrix is not None,
            "hybrid_loaded": user_content_profiles is not None,
            "movies_rows": len(movies_df),
        },
    }

@app.get("/recommend/{user_id}")
async def recommend_cf(user_id: int, top_n: int = 10):
    if user_embeddings is None or movie_embeddings is None:
        raise HTTPException(status_code=503, detail="CF models not loaded")
        
    user_idx = user2idx.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail="User ID not found in CF model")

    user_vec = user_embeddings[user_idx]
    scores = np.dot(movie_embeddings, user_vec)
    top_indices = np.argsort(scores)[-top_n:][::-1]

    recommended_movies = []
    for idx in top_indices:
        # idx is internal index, map back to movie_id
        if idx not in idx2movie: continue
        movie_id = idx2movie[idx]
        
        # Get title
        title_rows = movies_df.loc[movies_df["movieId"] == movie_id, "title"]
        movie_title = title_rows.values[0] if not title_rows.empty else "Unknown"

        recommended_movies.append({
            "movie_id": int(movie_id),
            "title": str(movie_title),
            "score": float(scores[idx])
        })

    return {"user_id": int(user_id), "recommendations": recommended_movies}

@app.get("/recommend/content/")
async def recommend_content(
    movie_id: Optional[int] = Query(None),
    title: Optional[str] = Query(None),
    top_n: int = Query(10, ge=1, le=100),
):
    if tfidf_sim_matrix is None:
        raise HTTPException(status_code=503, detail="Content-based artifacts not loaded.")

    if movie_id is None and title is None:
        raise HTTPException(status_code=400, detail="Provide either movie_id or title.")

    idx = None
    if movie_id is not None:
        if movie_id not in movies_df["movieId"].values:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found.")
        # need the index in the tfidf matrix. 
        # Index should match.
        idx = int(movies_df.index[movies_df["movieId"] == movie_id][0])
    else:
        found_title = find_movie_by_title(title)
        if found_title is None:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found.")
        idx = int(movies_df.index[movies_df["title"] == found_title][0])

    sims = cosine_similarity(tfidf_sim_matrix[idx:idx+1], tfidf_sim_matrix).flatten()
    if len(tfidf_sim_matrix.shape) == 2 and tfidf_sim_matrix.shape[0] == tfidf_sim_matrix.shape[1]:
         sims = tfidf_sim_matrix[idx]
    else:
         # Fallback 
         pass

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

@app.get("/recommend/hybrid/{user_id}")
async def recommend_hybrid(user_id: int, top_n: int = 10, cf_weight: float = 0.5):
    """
    Weighted Hybrid:
    Score = cf_weight * CF_Score + (1 - cf_weight) * Content_Profile_Score
    """
    if user_embeddings is None or tfidf_matrix is None or user_content_profiles is None:
        raise HTTPException(status_code=503, detail="Hybrid models not fully loaded")

    # Get CF Scores
    user_idx = user2idx.get(user_id)
    cf_scores = np.zeros(len(movies_df))
    if user_idx is not None:
        user_vec = user_embeddings[user_idx]
        # Pre-compute mapping from movies_df index -> movie_id -> movie_embedding index
        pass
    
    cf_scores_aligned = np.zeros(len(movies_df))
    
    # Content Part
    content_scores_aligned = np.zeros(len(movies_df))
    user_profile = user_content_profiles.get(user_id)
    
    if user_profile is not None:
        # Compute cosine sim between user_profile and all movies
        # tfidf_matrix is (n_movies, n_features)
        # user_profile is (n_features,)
        # scores = tfidf_matrix dot user_profile
        content_scores_aligned = tfidf_matrix.dot(user_profile)
    
    # Fill CF Scores
    if user_idx is not None:

        pass


    for i, row in movies_df.iterrows():
        mid = row['movieId']
        
        # CF Score
        if user_idx is not None:
            mid_idx = movie2idx.get(mid) or movie2idx.get(str(mid))
            if mid_idx is not None:
                cf_scores_aligned[i] = np.dot(movie_embeddings[mid_idx], user_embeddings[user_idx])
            
    # Normalize scores to 0-1 range roughly?
    # CF scores are roughly ratings (1-5).
    # Content scores are cosine sim (0-1).
    # normalize CF
    cf_scores_aligned /= 5.0
    
    final_scores = (cf_weight * cf_scores_aligned) + ((1 - cf_weight) * content_scores_aligned)
    
    top_indices = np.argsort(final_scores)[-top_n:][::-1]
    
    recs = []
    for i in top_indices:
        recs.append({
            "movieId": int(movies_df.iloc[i]["movieId"]),
            "title": str(movies_df.iloc[i]["title"]),
            "score": float(final_scores[i]),
            "cf_component": float(cf_scores_aligned[i]),
            "content_component": float(content_scores_aligned[i])
        })
        
    return {"user_id": user_id, "recommendations": recs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
