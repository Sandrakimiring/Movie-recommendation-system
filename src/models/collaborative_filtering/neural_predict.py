import torch
import torch.nn as nn
import joblib
import pandas as pd
from pathlib import Path
import logging
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / 'neural_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)


class NeuralCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        logger.info(f"Initializing NeuralCF with num_users={num_users}, num_movies={num_movies}, embedding_dim={embedding_dim}")
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        logger.debug("NeuralCF layers created")

    def forward(self, user, movie):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Forward called with user tensor shape {getattr(user, 'shape', None)} and movie tensor shape {getattr(movie, 'shape', None)}")
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        x = torch.cat([u, m], dim=-1)
        out = self.fc_layers(x)
        squeezed = out.squeeze()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Forward output shape {getattr(squeezed, 'shape', None)}")
        return squeezed


def load_model_and_mappings():
    logger.info("Loading model mappings and weights")
    try:
        user2idx = joblib.load(MODEL_DIR / "user2idx.pkl")
        logger.info(f"Loaded user2idx mapping with {len(user2idx)} entries")
    except Exception as e:
        logger.exception("Failed to load user2idx mapping")
        raise

    try:
        movie2idx = joblib.load(MODEL_DIR / "movie2idx.pkl")
        logger.info(f"Loaded movie2idx mapping with {len(movie2idx)} entries")
    except Exception as e:
        logger.exception("Failed to load movie2idx mapping")
        raise

    model = NeuralCF(num_users=len(user2idx), num_movies=len(movie2idx))
    try:
        state_path = MODEL_DIR / "ncf_model.pt"
        logger.info(f"Loading model state from {state_path}")
        model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model state loaded and set to eval mode")
    except Exception as e:
        logger.exception("Failed to load model state")
        raise

    return model, user2idx, movie2idx


def predict_rating(model, user2idx, movie2idx, user_id, movie_id):
    logger.info(f"Predicting rating for user_id={user_id}, movie_id={movie_id}")
    if user_id not in user2idx:
        logger.error(f"User id {user_id} not found in user2idx mapping")
        raise ValueError("User or movie not found in mappings")
    if movie_id not in movie2idx:
        logger.error(f"Movie id {movie_id} not found in movie2idx mapping")
        raise ValueError("User or movie not found in mappings")

    u_idx = torch.tensor([user2idx[user_id]])
    m_idx = torch.tensor([movie2idx[movie_id]])
    logger.debug(f"Converted to tensors u_idx={u_idx.tolist()}, m_idx={m_idx.tolist()}")
    with torch.no_grad():
        rating = model(u_idx, m_idx).item()
    logger.info(f"Predicted rating for user {user_id}, movie {movie_id}: {rating:.2f}")
    return rating


def recommend_top_n(model, user2idx, movie2idx, user_id, n=10):
    logger.info(f"Generating top {n} recommendations for user_id={user_id}")
    if user_id not in user2idx:
        logger.error(f"User id {user_id} not found in user2idx mapping")
        raise ValueError("User not found in mappings")

    u_idx = torch.tensor([user2idx[user_id]])
    model.eval()

    predictions = []
    total_movies = len(movie2idx)
    logger.info(f"Scoring {total_movies} movies for user {user_id}")
    with torch.no_grad():
        for i, (movie_id, m_idx_val) in enumerate(movie2idx.items(), start=1):
            m_tensor = torch.tensor([m_idx_val])
            rating = model(u_idx, m_tensor).item()
            predictions.append((movie_id, rating))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Scored movie {movie_id} ({i}/{total_movies}): {rating:.4f}")
            # occasional progress info for large sets
            if i % 1000 == 0:
                logger.info(f"Scored {i}/{total_movies} movies")

    logger.info(f"Completed scoring. Total predictions: {len(predictions)}")
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    logger.info(f"Returning top {len(top_movies)} movies for user {user_id}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Top movies: {top_movies[:5]}")
    return top_movies


if __name__ == "__main__":
    logger.info("Starting neural_predict script")
    try:
        model, user2idx, movie2idx = load_model_and_mappings()
    except Exception:
        logger.error("Exiting due to load failure")
        raise

    try:
        movies_df = pd.read_csv(DATA_DIR / "movies.csv")
        logger.info(f"Loaded movies dataframe with {len(movies_df)} rows from {DATA_DIR / 'movies.csv'}")
    except Exception:
        logger.exception("Failed to load movies.csv")
        raise

    user_id = 6
    logger.info(f"Requesting recommendations for user {user_id}")
    try:
        top_movies = recommend_top_n(model, user2idx, movie2idx, user_id, n=10)
    except Exception:
        logger.exception("Failed to generate recommendations")
        raise

    for mid, score in top_movies:
        try:
            title_vals = movies_df.loc[movies_df['movieId'] == mid, 'title'].values
            if len(title_vals) == 0:
                logger.warning(f"Title not found for movieId {mid}")
                title = "Unknown Title"
            else:
                title = title_vals[0]
        except Exception:
            logger.exception(f"Failed to lookup title for movieId {mid}")
            title = "Unknown Title"

        line = f"{title} ({mid}) — predicted score: {score:.2f}"
        print(line)
    