import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

try:
    from src.models.collaborative_filtering.svd_cf import load_ratings, train_svd, save_model
except ImportError:
    load_ratings = None
    train_svd = None
    save_model = None
    logging.warning("scikit-surprise not found. SVD training will be skipped.")
from src.models.content_based.tfidf_train import train_tfidf_model
from src.models.hybrid.train_hybrid import train_hybrid_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Starting Master Training Pipeline ===")
    
    #Train Collaborative Filtering (SVD)
    if train_svd:
        try:
            logger.info("--- Step 1: Collaborative Filtering (SVD) ---")
            ratings_df = load_ratings()
            svd_model, trainset, rmse = train_svd(ratings_df)
            save_model(svd_model, trainset)
        except Exception as e:
            logger.error(f"SVD Training failed: {e}")
    else:
        logger.warning("Skipping SVD training (module not found).")
    
    #Train Content-Based (TF-IDF)
    try:
        logger.info("--- Step 2: Content-Based (TF-IDF) ---")
        train_tfidf_model()
    except Exception as e:
        logger.error(f"TF-IDF Training failed: {e}")

    #Train Hybrid (User Profiles)
    try:
        logger.info("--- Step 3: Hybrid Model (User Profiles) ---")
        train_hybrid_model()
    except Exception as e:
        logger.error(f"Hybrid Training failed: {e}")

    logger.info("=== Training Pipeline Completed ===")

if __name__ == "__main__":
    main()
