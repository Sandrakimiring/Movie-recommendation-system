import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from surprise import SVD, Dataset, Reader
import logging


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=LOGS_DIR / 'cf_model_ev.log',
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
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user, movie):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        x = torch.cat([u, m], dim=-1)
        return self.fc_layers(x).squeeze()

def load_svd_model():
    logger.info("Loading SVD model...")
    svd = joblib.load(MODEL_DIR / "svd_model.pkl")
    logger.info("SVD model loaded.")
    return svd

def load_ncf_model():
    logger.info("Loading Neural CF model and mappings...")
    user2idx = joblib.load(MODEL_DIR / "user2idx.pkl")
    movie2idx = joblib.load(MODEL_DIR / "movie2idx.pkl")
    model = NeuralCF(num_users=len(user2idx), num_movies=len(movie2idx))
    model.load_state_dict(torch.load(MODEL_DIR / "ncf_model.pt"))
    model.eval()
    logger.info(f"Neural CF model loaded: {len(user2idx)} users, {len(movie2idx)} movies")
    return model, user2idx, movie2idx

# Evaluation metrics
def precision_recall_at_k(true_ratings, pred_ratings, k=10, threshold=4.0):
    top_k_idx = np.argsort(pred_ratings)[::-1][:k]
    top_k_true = np.array(true_ratings)[top_k_idx]
    hits = np.sum(top_k_true >= threshold)
    precision = hits / k
    recall = hits / np.sum(np.array(true_ratings) >= threshold)
    return precision, recall

def ndcg_at_k(true_ratings, pred_ratings, k=10):
    order = np.argsort(pred_ratings)[::-1][:k]
    rel = np.take(true_ratings, order)
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal_rel = np.sort(true_ratings)[::-1][:k]
    idcg = np.sum((2**ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

# Main evaluation function
def evaluate_models():
    ratings_csv = DATA_DIR / "ratings.csv"
    df = pd.read_csv(ratings_csv)
    logger.info(f"Loaded ratings data: {df.shape[0]} rows")

    # Load models
    svd = load_svd_model()
    ncf_model, user2idx, movie2idx = load_ncf_model()

    users = df.userId.unique()
    logger.info(f"Evaluating top 200 users for speed...")

    # Containers
    svd_rmse_list, svd_mae_list, svd_prec_list, svd_rec_list, svd_ndcg_list = [], [], [], [], []
    ncf_rmse_list, ncf_mae_list, ncf_prec_list, ncf_rec_list, ncf_ndcg_list = [], [], [], [], []

    for user in users[:200]:
        user_ratings = df[df.userId == user]
        true_r = user_ratings['rating'].values
        movie_ids = user_ratings['movieId'].values

        #  SVD
        svd_preds = [svd.predict(user, mid).est for mid in movie_ids]
        svd_preds = np.array(svd_preds)
        svd_rmse_list.append(np.sqrt(np.mean((svd_preds - true_r)**2)))
        svd_mae_list.append(np.mean(np.abs(svd_preds - true_r)))
        p, r = precision_recall_at_k(true_r, svd_preds)
        svd_prec_list.append(p)
        svd_rec_list.append(r)
        svd_ndcg_list.append(ndcg_at_k(true_r, svd_preds))

        # Neural CF 
        ncf_preds = []
        for mid in movie_ids:
            if user not in user2idx or mid not in movie2idx:
                continue
            u_idx = torch.tensor([user2idx[user]])
            m_idx = torch.tensor([movie2idx[mid]])
            with torch.no_grad():
                rating = ncf_model(u_idx, m_idx).item()
            ncf_preds.append(rating)
        if len(ncf_preds) == 0:
            continue
        ncf_preds = np.array(ncf_preds)
        ncf_rmse_list.append(np.sqrt(np.mean((ncf_preds - true_r[:len(ncf_preds)])**2)))
        ncf_mae_list.append(np.mean(np.abs(ncf_preds - true_r[:len(ncf_preds)])))
        p, r = precision_recall_at_k(true_r[:len(ncf_preds)], ncf_preds)
        ncf_prec_list.append(p)
        ncf_rec_list.append(r)
        ncf_ndcg_list.append(ndcg_at_k(true_r[:len(ncf_preds)], ncf_preds))

    logger.info("------ SVD Metrics ------")
    logger.info(f"RMSE: {np.mean(svd_rmse_list):.4f}")
    logger.info(f"MAE: {np.mean(svd_mae_list):.4f}")
    logger.info(f"Precision@10: {np.mean(svd_prec_list):.4f}")
    logger.info(f"Recall@10: {np.mean(svd_rec_list):.4f}")
    logger.info(f"NDCG@10: {np.mean(svd_ndcg_list):.4f}")

    logger.info("------ Neural CF Metrics ------")
    logger.info(f"RMSE: {np.mean(ncf_rmse_list):.4f}")
    logger.info(f"MAE: {np.mean(ncf_mae_list):.4f}")
    logger.info(f"Precision@10: {np.mean(ncf_prec_list):.4f}")
    logger.info(f"Recall@10: {np.mean(ncf_rec_list):.4f}")
    logger.info(f"NDCG@10: {np.mean(ncf_ndcg_list):.4f}")

if __name__ == "__main__":
    evaluate_models()
