import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / 'tfidf_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

from difflib import get_close_matches

def find_movie(title, movies_df):
    title = title.strip().lower()
    movies_df['title'] = movies_df['title'].str.strip().str.lower()
    matches = get_close_matches(title, movies_df['title'], n=1, cutoff=0.6)
    return matches[0] if matches else None



def load_model_and_data():
    """Load TF-IDF model, matrix, and metadata."""
    logging.info("Loading TF-IDF model and matrix...")
    tfidf_model = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(MODEL_DIR / "tfidf_similarity_matrix.pkl")
   
    logging.info("Loading movies metadata...")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    return tfidf_model, tfidf_matrix, movies_df


def get_recommendations(title, tfidf_model, tfidf_matrix, movies_df, top_n=10):
    """Return top-N content-based recommendations for a given movie title."""

    if title not in movies_df["title"].values:
        logging.warning(f"Movie '{title}' not found in dataset.")
        return []

    idx = movies_df.index[movies_df["title"] == title][0]

    # Directly use similarity row 
    cosine_similarities = tfidf_matrix[idx]

    similar_indices = cosine_similarities.argsort()[-(top_n + 1):][::-1]
    similar_scores = cosine_similarities[similar_indices]

    recommendations = [
        (movies_df.iloc[i]["title"], round(similar_scores[pos], 3))
        for pos, i in enumerate(similar_indices)
        if i != idx
    ][:top_n]

    logging.info(f"Top {top_n} recommendations for '{title}': {recommendations}")
    return recommendations



if __name__ == "__main__":
    logging.info("Starting TF-IDF prediction...")
    tfidf_model, tfidf_matrix, movies_df = load_model_and_data()

    user_input = input("Enter a movie title: ")

    movie_title = find_movie(user_input, movies_df)
    if movie_title:
        recommendations = get_recommendations(movie_title, tfidf_model, tfidf_matrix, movies_df)
        for rec_title, score in recommendations:
            print(f"{rec_title} (Similarity: {score})")
    else:
        logging.warning(f"No close match found for movie title '{user_input}'.")

    logging.info("Prediction completed successfully.")
