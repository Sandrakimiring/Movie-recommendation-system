import pandas as pd
import numpy as np
import logging
import sys
import mlflow 
import mlflow.sklearn
from pathlib import Path
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pickle
import os
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = BASE_DIR / 'logs'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models' 
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / 'svd_cf.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_ratings():
    logging.info("Loading ratings data")
    df = pd.read_csv(PROCESSED_DATA_DIR / 'ratings.csv')
    logging.info(f"Ratings data shape: {df.shape}")
    return df

def train_svd(ratings_df, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
    logging.info("Preparing dataset for Surprise library...")
    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2)
    
    logging.info("Training SVD model...")
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)
    
    logging.info("Evaluating SVD model...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    logging.info(f"SVD RMSE on test set: {rmse:.4f}")
    
    return model, trainset

def save_model(model, model_name='svd_model.pkl'):
    joblib.dump(model, MODEL_DIR / model_name)
    logging.info(f"Model saved to {MODEL_DIR / model_name}")
    
    user_factors = np.array([model.pu[i] for i in range(model.pu.shape[0])])
    movie_factors = np.array([model.qi[i] for i in range(model.qi.shape[0])])

    np.save(MODEL_DIR / 'user_embeddings.npy', user_factors)
    np.save(MODEL_DIR / 'movie_embeddings.npy', movie_factors)
    logging.info("User and movie embeddings saved.")
    
if __name__ == "__main__":
    try:
        ratings_df = load_ratings()
        svd_model, trainset = train_svd(ratings_df)
        save_model(svd_model, trainset)
        logging.info("SVD Collaborative Filtering model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")