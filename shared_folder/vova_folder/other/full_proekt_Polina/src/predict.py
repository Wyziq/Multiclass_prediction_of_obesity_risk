
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import load_joblib, project_root
from .data_preprocessing import clean_base, TARGET_COL


def predict_csv(model_path: str | Path, input_csv: str | Path, output_csv: str | Path):
    model = load_joblib(model_path)

    df = pd.read_csv(input_csv)
    df = clean_base(df)

    # если вдруг таргет есть в файле — убираем
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    preds = model.predict(df)
    out = pd.DataFrame({"prediction": preds})
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(project_root() / "models" / "best_model.joblib"))
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(project_root() / "reports" / "predictions.csv"))
    args = parser.parse_args()

    out = predict_csv(args.model, args.input, args.output)
    print("Saved predictions to:", out)


if __name__ == "__main__":
    main()
