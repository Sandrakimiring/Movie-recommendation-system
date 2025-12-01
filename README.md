Movie-Recommendation-System
==============================

End-to-End Netflix Recommender System

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## How to Run

### Prerequisites
- Python 3.9+
- Docker (optional)

### Local Setup (Windows)
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application**:
   Double-click `run.bat` or run in terminal:
   ```cmd
   run.bat
   ```
   This script will:
   - Train the models (if needed).
   - Start the API server.

3. **Access the API**:
   Open your browser to [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API documentation.

### Docker Setup
1. **Build the Image**:
   ```bash
   docker build -t movie-recs .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 8000:8000 movie-recs
   ```

## Key Files Explained

- **`main.py`**: The entry point of the application. It starts the Uvicorn server which hosts the FastAPI app.
- **`src/api/app.py`**: The core API logic. It defines the endpoints (`/recommend/...`), loads the trained models, and handles user requests.
- **`src/models/train_model.py`**: The master training script. It orchestrates the training of:
    - **Collaborative Filtering**: SVD model (users similar to you).
    - **Content-Based**: TF-IDF model (movies similar to this movie).
    - **Hybrid**: User Profiles (your taste based on movie content).
- **`src/models/hybrid/train_hybrid.py`**: Logic for generating User Content Profiles for the hybrid model.
- **`Dockerfile`**: Configuration for packaging the app into a container.
- **`run.bat`**: A helper script for Windows users to easily train and run the app.

## API Endpoints

- **`GET /`**: Health check and model status.
- **`GET /recommend/{user_id}`**: Get recommendations based on Collaborative Filtering (what similar users liked).
- **`GET /recommend/content/?title=...`**: Get recommendations based on Content-Based Filtering (movies similar to the query).
- **`GET /recommend/hybrid/{user_id}`**: Get recommendations using a weighted hybrid approach (combining user taste and similar users).

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
