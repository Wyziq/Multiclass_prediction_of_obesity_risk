
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .data_preprocessing import load_data, clean_base, make_variant_tables, split_xy, build_pipeline_for_training, DatasetVariants
from .utils import evaluate_cv, save_joblib, save_metrics, project_root, set_seed


def candidate_models(seed: int = 42):
    """Лучшие базовые модели из того, что обычно уже делали в проекте:
    - LogisticRegression (хорошая база + интерпретируемость)
    - RandomForest (сильный baseline для табличных)
    - SVC RBF (часто даёт топ на этом датасете, но тяжелее)
    """
    return {
        "logreg": LogisticRegression(max_iter=3000, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", probability=False),
    }


def select_best_model(df_variant: pd.DataFrame, seed: int = 42):
    X, y = split_xy(df_variant)
    results = {}

    for name, model in candidate_models(seed).items():
        # Для SVM и логрег — scaling полезен; для RF не обязателен, но оставим (не мешает).
        pipe = build_pipeline_for_training(df_variant, model=model, onehot_nominal=True, scale_numeric=True)
        metrics = evaluate_cv(pipe, X, y, cv_splits=5, seed=seed)
        results[name] = metrics

    # выбираем по f1_macro
    best_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
    return best_name, results


def train_best(
    raw_csv: str | Path,
    variant_name: str = DatasetVariants.full_with_height_weight,
    seed: int = 42,
):
    set_seed(seed)

    df_raw = load_data(raw_csv)
    variants = make_variant_tables(df_raw)
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. Available: {list(variants.keys())}")

    df_v = variants[variant_name]

    best_name, cv_results = select_best_model(df_v, seed=seed)

    # финальное обучение: делаем holdout test только для финального отчёта (не для подбора!)
    X, y = split_xy(df_v)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = candidate_models(seed)[best_name]
    pipe = build_pipeline_for_training(df_v, model=model, onehot_nominal=True, scale_numeric=True)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    root = project_root()
    model_path = save_joblib(pipe, root / "models" / "best_model.joblib")
    save_metrics({"variant": variant_name, "best_model": best_name, "cv": cv_results, "holdout_report": report},
                 root / "reports" / "metrics.json")

    return best_name, cv_results, model_path


if __name__ == "__main__":
    root = project_root()
    raw_path = root / "data" / "raw" / "ObesityDataSet.csv"
    best_name, cv_results, model_path = train_best(raw_path, DatasetVariants.full_with_height_weight, seed=42)
    print("Best:", best_name)
    print("Saved model to:", model_path)
