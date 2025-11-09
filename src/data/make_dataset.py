import pandas as pd
from pathlib import Path
import logging
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
LOGS_PATH = BASE_DIR / "logs"

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=LOGS_PATH / "data_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # append to existing log
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def load_ratings():
    """Load ratings.dat"""
    logging.info("Loading ratings.dat...")
    df = pd.read_csv(
        RAW_DATA_PATH / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    logging.info(f"Ratings loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_movies():
    """Load movies.dat"""
    logging.info("Loading movies.dat...")
    df = pd.read_csv(
        RAW_DATA_PATH / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )
    logging.info(f"Movies loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_users():
    """Load users.dat"""
    logging.info("Loading users.dat...")
    df = pd.read_csv(
        RAW_DATA_PATH / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )
    logging.info(f"Users loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_csv(df, filename):
    """Save processed dataframe to CSV"""
    df.to_csv(PROCESSED_DATA_PATH / filename, index=False)
    logging.info(f"Saved {filename} to processed folder!")

if __name__ == "__main__":
    logging.info("Starting data pipeline...")

    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    save_csv(ratings, "ratings.csv")
    save_csv(movies, "movies.csv")
    save_csv(users, "users.csv")

    logging.info("Data pipeline finished successfully!")
