import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.pytorch

# --------------------------------------------------------------------- #
# 1. Modal app + GPU image
# --------------------------------------------------------------------- #
app = modal.App("neural-cf-movie-rec")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "numpy", "scikit-learn", "mlflow")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
)

# --------------------------------------------------------------------- #
# 2. Paths (project-root relative – works both locally and on Modal)
# --------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "logs"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
# 3. Logging (writes to a file *and* prints to stdout)
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH / "neural_cf_training.log"),
        logging.StreamHandler()
    ]
)

# --------------------------------------------------------------------- #
# 4. Dataset (unchanged)
# --------------------------------------------------------------------- #
class RatingsDataset(Dataset):
    def __init__(self, ratings_df):
        self.user_ids = torch.tensor(ratings_df.userId.values, dtype=torch.long)
        self.movie_ids = torch.tensor(ratings_df.movieId.values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df.rating.values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

# --------------------------------------------------------------------- #
# 5. Neural CF model (unchanged)
# --------------------------------------------------------------------- #
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        x = torch.cat([user_emb, movie_emb], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# --------------------------------------------------------------------- #
# 6. Training helper (unchanged)
# --------------------------------------------------------------------- #
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for user, movie, rating in dataloader:
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)

            optimizer.zero_grad()
            pred = model(user, movie)
            loss = criterion(pred, rating)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * rating.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return model

# --------------------------------------------------------------------- #
# 7. Save model + embeddings (unchanged)
# --------------------------------------------------------------------- #
def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH / "neural_cf_model.pt")
    logging.info("Saved neural CF model.")

    user_emb = model.user_embedding.weight.data.cpu().numpy()
    movie_emb = model.movie_embedding.weight.data.cpu().numpy()
    np.save(MODEL_PATH / "user_embeddings_nn.npy", user_emb)
    np.save(MODEL_PATH / "movie_embeddings_nn.npy", movie_emb)
    logging.info("Saved user & movie embeddings (Neural CF).")

# --------------------------------------------------------------------- #
# 8. Modal GPU function – this runs on the cloud
# --------------------------------------------------------------------- #
@app.function(image=image, gpu="A100", timeout=3600)   # 1-hour timeout
def run_neural_cf():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Modal function using device: {device}")

    # ----------------------------------------------------------------- #
    # Load data
    # ----------------------------------------------------------------- #
    ratings_path = RAW_DATA_PATH / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found at {ratings_path}")

    ratings_df = pd.read_csv(ratings_path)
    logging.info(f"Loaded {len(ratings_df):,} ratings")

    num_users = ratings_df.userId.max()
    num_movies = ratings_df.movieId.max()

    # ----------------------------------------------------------------- #
    # DataLoader
    # ----------------------------------------------------------------- #
    dataset = RatingsDataset(ratings_df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # ----------------------------------------------------------------- #
    # Model + training
    # ----------------------------------------------------------------- #
    model = NeuralCF(num_users, num_movies, embedding_dim=50).to(device)

    with mlflow.start_run():
        model = train_model(model, dataloader, epochs=10, lr=0.001)

        # Log hyper-params
        mlflow.log_params({
            "embedding_dim": 50,
            "epochs": 10,
            "lr": 0.001,
            "batch_size": 1024
        })

        # Save & log model
        save_model(model)
        mlflow.pytorch.log_model(model, "model")

    return "Neural CF training completed!"

# --------------------------------------------------------------------- #
# 9. Local entrypoint – what you call with `modal run`
# --------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    print("Launching Neural CF on Modal GPU...")
    result = run_neural_cf.remote()
    print(result)
