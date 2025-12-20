"""Predict using the saved model.

Example:
    python -m src.predict --input data/raw/ObesityDataSet.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import make_feature_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="predictions.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train first: python -m src.train --model rf")

    pipe = joblib.load(model_path)

    df = pd.read_csv(args.input)
    X, _ = make_feature_target(df, allow_missing_target=True)

    preds = pipe.predict(X)
    pd.DataFrame({"prediction": preds}).to_csv(args.output, index=False)
    print("Saved predictions to:", args.output)


if __name__ == "__main__":
    main()
