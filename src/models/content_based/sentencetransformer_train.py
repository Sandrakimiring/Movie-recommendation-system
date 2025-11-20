import logging
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / 'sentence_transformer_train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)


def train_sentence_transformer():
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

    logger.info("Preparing training data for SentenceTransformer...")
    train_examples = [InputExample(texts=[text]) for text in movies_df["text"]]
    logger.info(f"Prepared {len(train_examples)} text samples for training.")

    logger.info("Initializing SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Training SentenceTransformer model (epoch=1)...")
    train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    logger.info("SentenceTransformer model training complete.")

    logger.info("Saving trained SentenceTransformer model...")
    model.save(MODEL_DIR / "sentence_transformer_model")
    logger.info("SentenceTransformer model saved successfully.")


if __name__ == "__main__":
    train_sentence_transformer()
