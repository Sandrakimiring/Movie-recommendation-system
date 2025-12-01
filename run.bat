@echo off
echo Starting Movie Recommendation System...
python src/models/train_model.py
if %errorlevel% neq 0 (
    echo Training failed!
    exit /b %errorlevel%
)
echo Training complete. Starting API...
python main.py
