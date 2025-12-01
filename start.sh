#!/bin/bash
echo "Starting Movie Recommendation System..."
# Check if models exist, if not train
if [ ! -f "models/user_content_profiles.pkl" ]; then
    echo "Models not found. Training..."
    python src/models/train_model.py
fi
echo "Starting API..."
python main.py
