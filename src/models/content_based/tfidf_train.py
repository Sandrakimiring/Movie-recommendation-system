import logging
from pathlib import Path
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / 'tfidf_train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)


def train_tfidf_model():
    logger.info("Loading movie data...")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    logger.info(f"Movie data loaded with columns: {list(movies_df.columns)}")

    text_column = None
    for col in ["description", "overview", "plot", "storyline"]:
        if col in movies_df.columns:
            text_column = col
            logger.info(f"Using '{col}' as the text description column.")
            break

    if not text_column:
        logger.warning("No description-like column found. Combining title + genres instead.")
        movies_df["text"] = (
            movies_df["title"].fillna('') + " " +
            movies_df.get("genres", "").fillna('')
        )
    else:
        movies_df["text"] = movies_df[text_column].fillna('')

    def clean_text(text):
        return re.sub(r'\s*\(\d{4}\)|\s*\d{4}', '', str(text))

    movies_df["text"] = movies_df["text"].apply(clean_text)
    logger.info("Cleaned dates from text data to prevent year-bias.")
    
    logger.info("Training TF-IDF model...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df["text"])
    logger.info("TF-IDF model trained successfully.")

    logger.info("Computing cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    logger.info("Saving TF-IDF model and cosine similarity matrix...")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_matrix, MODEL_DIR / "tfidf_matrix.pkl")
    joblib.dump(cosine_sim, MODEL_DIR / "tfidf_similarity_matrix.pkl")
    logger.info("TF-IDF model and similarity matrix saved successfully.")

if __name__ == "__main__":
    train_tfidf_model()
