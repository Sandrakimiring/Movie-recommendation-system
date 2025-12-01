import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

logging.basicConfig(
    filename=LOGS_DIR / 'hybrid_train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

def train_hybrid_model():
    logger.info("Starting Hybrid Model training (User Profile Generation)...")
    
    # 1. Load Data
    try:
        ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
        movies_df = pd.read_csv(DATA_DIR / "movies.csv")
        tfidf_matrix = joblib.load(MODEL_DIR / "tfidf_matrix.pkl")
        logger.info(f"Loaded data: ratings={len(ratings_df)}, movies={len(movies_df)}")
    except Exception as e:
        logger.error(f"Failed to load data or artifacts: {e}")
        return
# Map movieId to Matrix Index
# index of movies_df corresponds to the index in tfidf_matrix.
    
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["movieId"])}
    
    #Compute User Profiles
    user_profiles = {}
    
    # Weighted average is good.
    # Profile = (Sum(Rating * MovieVec)) / Sum(Rating)
    
    grouped = ratings_df.groupby("userId")
    
    logger.info(f"Computing profiles for {len(grouped)} users...")
    
    count = 0
    for user_id, group in grouped:
        user_vector_sum = None
        weight_sum = 0.0
        
        for _, row in group.iterrows():
            mid = row["movieId"]
            rating = row["rating"]
            
            if mid in movie_id_to_idx:
                idx = movie_id_to_idx[mid]
                vec = tfidf_matrix[idx].toarray()[0]
                
                if user_vector_sum is None:
                    user_vector_sum = np.zeros_like(vec)
                
                user_vector_sum += vec * rating
                weight_sum += rating
        
        if user_vector_sum is not None and weight_sum > 0:
            user_profiles[user_id] = user_vector_sum / weight_sum
        
        count += 1
        if count % 100 == 0:
            logger.info(f"Processed {count} users...")

    # Save Profiles
    output_path = MODEL_DIR / "user_content_profiles.pkl"
    joblib.dump(user_profiles, output_path)
    logger.info(f"Saved {len(user_profiles)} user profiles to {output_path}")

if __name__ == "__main__":
    train_hybrid_model()
