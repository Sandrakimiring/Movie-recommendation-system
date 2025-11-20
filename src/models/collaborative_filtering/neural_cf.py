import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from pathlib import Path
import logging
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=LOGS_DIR / 'neural_cf.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

class RatingsDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

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
        out = self.fc_layers(x)
        return out.squeeze()


def train_model(ratings_csv: Path):
    df = pd.read_csv(ratings_csv)
    # Convert user/movie ids to continuous integers
    user2idx = {uid: i for i, uid in enumerate(df.userId.unique())}
    movie2idx = {mid: i for i, mid in enumerate(df.movieId.unique())}
    df['user_idx'] = df['userId'].map(user2idx)
    df['movie_idx'] = df['movieId'].map(movie2idx)

    dataset = RatingsDataset(df['user_idx'], df['movie_idx'], df['rating'])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = NeuralCF(num_users=len(user2idx), num_movies=len(movie2idx))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, m, r in loader:
            optimizer.zero_grad()
            pred = model(u, m)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")


    # Save model and mappings
    torch.save(model.state_dict(), MODEL_DIR / "ncf_model.pt")
    joblib.dump(user2idx, MODEL_DIR / "user2idx.pkl")
    joblib.dump(movie2idx, MODEL_DIR / "movie2idx.pkl")
    logging.info("NCF model and mappings saved successfully.")

if __name__ == "__main__":
    ratings_csv = BASE_DIR / "data" / "processed" / "ratings.csv"
    train_model(ratings_csv)
