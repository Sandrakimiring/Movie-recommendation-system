import joblib
from pathlib import Path
import logging
import numpy as np
from typing import Dict, List, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
MOVIES_DIR = BASE_DIR / 'data' / 'processed'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=LOGS_DIR / 'svd_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(_console)
logger = logging.getLogger(__name__)


def _norm_id(x: Any) -> str:
    """Normalize ID to string."""
    return str(x)

def load_model_and_embeddings(
    model_path: Path = MODEL_DIR / "svd_model.pkl",
    user_emb_path: Path = MODEL_DIR / "user_embeddings.pkl",
    movie_emb_path: Path = MODEL_DIR / "movie_embeddings.pkl",
) -> Tuple[Any, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load trained SVD model and embeddings."""
    logger.info(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)
    logger.info("Model loaded.")

    logger.info(f"Loading user embeddings from {user_emb_path} ...")
    user_map = joblib.load(user_emb_path) if user_emb_path.exists() else {}
    logger.info(f"Loaded {len(user_map)} user embeddings.")

    logger.info(f"Loading movie embeddings from {movie_emb_path} ...")
    movie_map = joblib.load(movie_emb_path) if movie_emb_path.exists() else {}
    logger.info(f"Loaded {len(movie_map)} movie embeddings.")

    return model, user_map, movie_map

def load_movie_titles(movies_csv: Path = MOVIES_DIR / "movies.csv") -> Dict[str, str]:
    """Load movieId -> title mapping."""
    import pandas as pd
    if not movies_csv.exists():
        logger.warning(f"Movies CSV not found at {movies_csv}. Titles will not be available.")
        return {}
    df = pd.read_csv(movies_csv)
    df["movieId"] = df["movieId"].astype(str)
    mapping = dict(zip(df["movieId"], df["title"]))
    logger.info(f"Loaded {len(mapping)} movie titles from {movies_csv}.")
    return mapping

def recommend_top_n(model, user_map: Dict[str, np.ndarray], movie_map: Dict[str, np.ndarray],
                    user_id: Any, n: int = 10) -> List[Tuple[str, float]]:
    """Recommend top-n movies for a user."""
    u = _norm_id(user_id)

    # Use embeddings if available
    if u in user_map and len(movie_map) > 0:
        user_vec = np.asarray(user_map[u], dtype=float)
        movie_ids, movie_vecs = [], []
        for mid, vec in movie_map.items():
            movie_ids.append(mid)
            movie_vecs.append(np.asarray(vec, dtype=float))
        movie_vecs = np.vstack(movie_vecs)
        scores = movie_vecs.dot(user_vec)
        top_idx = np.argsort(scores)[-n:][::-1]
        recommendations = [(movie_ids[i], float(scores[i])) for i in top_idx]
        logger.info(f"Top-{n} recommendations computed via embeddings for user {u}")
        return recommendations

    # Fallback: model.predict (slower)
    logger.info("Falling back to model.predict over all movies.")
    scored = [(mid, float(model.predict(u, mid).est)) for mid in movie_map.keys()]
    top_n = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
    return top_n


if __name__ == "__main__":
    USER_ID = 6      
    TOP_N = 10       
    model, user_map, movie_map = load_model_and_embeddings()
    movie_titles = load_movie_titles()

    logger.info(f"Loaded model and embeddings: {len(user_map)} users, {len(movie_map)} movies")
    logger.info(f"Loaded {len(movie_titles)} movie titles")

    recommendations = recommend_top_n(model, user_map, movie_map, USER_ID, n=TOP_N)
    
    print(f"\nTop {TOP_N} recommendations for user {USER_ID}:")
    for mid, score in recommendations:
        title = movie_titles.get(str(mid), None)
        if title:
            print(f"  {title} ({mid}) — score {score:.4f}")
        else:
            print(f"  {mid} — score {score:.4f}")
