"""Training entry point.

Examples (from project root):
    python -m src.train --model rf --variant 01_full_with_height_weight
    python -m src.train --model catboost --variant 02_with_bmi

Important:
- Model selection is done via cross-validation on TRAIN.
- Test split is used only once for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    build_preprocessor,
    load_raw_dataset,
    make_feature_target,
    save_dataset_variants,
)
from src.models_registry import get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="logreg", help="logreg | svm | rf | catboost")
    p.add_argument("--variant", type=str, default="01_full_with_height_weight")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--scoring", type=str, default="f1_macro")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    raw_path = root / "data" / "raw" / "ObesityDataSet.csv"
    processed_dir = root / "data" / "processed"

    df = load_raw_dataset(raw_path)
    variants = save_dataset_variants(df, processed_dir)

    if args.variant not in variants:
        raise ValueError(f"Unknown variant '{args.variant}'. Available: {sorted(variants.keys())}")

    X, y = make_feature_target(variants[args.variant])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipe = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", get_model(args.model)),
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=args.cv, scoring=args.scoring)
    cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True)

    (root / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, root / "models" / "best_model.joblib")

    (root / "reports").mkdir(exist_ok=True)
    metrics = {
        "model": args.model,
        "variant": args.variant,
        "cv": {"folds": args.cv, "scoring": args.scoring, "mean": cv_mean, "std": cv_std},
        "test": {"f1_macro": test_f1, "classification_report": report},
    }
    (root / "reports" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"CV {args.scoring}: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Test f1_macro: {test_f1:.4f}")
    print("Saved: models/best_model.joblib and reports/metrics.json")


if __name__ == "__main__":
    main()
