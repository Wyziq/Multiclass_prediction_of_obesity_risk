# Added models and wiring

Added model notebooks:
- notebooks/legacy_models/model_Random_Forest.ipynb
- notebooks/legacy_models/CatBoost.ipynb

Added code wiring:
- src/models_registry.py
- src/train.py (CV on train, saves models/best_model.joblib)
- src/predict.py

Run examples:
- python -m src.train --model rf --variant 01_full_with_height_weight
- python -m src.train --model catboost --variant 02_with_bmi   (needs: pip install catboost)
