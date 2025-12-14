
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


TARGET_COL = "NObeyesdad"

# Эти признаки в исходном датасете числовые, но по смыслу категориальные/дискретные.
# По требованию команды: округляем и приводим к int.
DISCRETE_CATEGORICAL = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]

BINARY_YES_NO = [
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
]

# Ординальные категориальные признаки: есть естественный порядок.
ORDINAL_MAPS: Dict[str, Dict[str, int]] = {
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
}

# Номинальные категориальные (порядка нет) — лучше one-hot.
NOMINAL_CATEGORICAL = ["Gender", "MTRANS"]


@dataclass(frozen=True)
class DatasetVariants:
    full_with_height_weight: str = "01_full_with_height_weight.csv"
    with_bmi: str = "02_with_bmi.csv"
    no_height: str = "03_no_height.csv"
    no_weight: str = "04_no_weight.csv"
    no_height_no_weight: str = "05_no_height_no_weight.csv"
    only_height_weight: str = "06_only_height_weight.csv"


def load_data(path: str | Path) -> pd.DataFrame:
    """Загрузка исходного датасета."""
    return pd.read_csv(path)


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет BMI = Weight / Height^2 (Height в метрах)."""
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df


def _round_discrete_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DISCRETE_CATEGORICAL:
        if col in df.columns:
            df[col] = np.rint(df[col]).astype(int)
    return df


def _encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {"no": 0, "yes": 1}
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)
    return df


def _encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mp in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mp).astype(int)
    return df


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Единая базовая обработка:
    - округление дискретных категориальных
    - yes/no -> 0/1
    - ординальные маппинги
    """
    out = df.copy()
    out = _round_discrete_features(out)
    out = _encode_binary_yes_no(out)
    out = _encode_ordinal(out)
    return out


def split_xy(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(
    X: pd.DataFrame,
    *,
    onehot_nominal: bool = True,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Строит ColumnTransformer.

    Логика кодирования:
    - бинарные yes/no и ординальные уже преобразованы в числа (clean_base)
    - дискретные категориальные (FCVC...) уже int (clean_base)
    - номинальные (Gender, MTRANS) по умолчанию one-hot (лучше для линейных/SVM)
    """
    # numeric = все числовые, кроме номинальных object
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if num_cols:
        if scale_numeric:
            transformers.append(("num", StandardScaler(), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

    if cat_cols:
        if onehot_nominal:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        else:
            # fallback: просто пропускаем (не рекомендуется)
            transformers.append(("cat", "passthrough", cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_variant_tables(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Готовит 6 вариантов датасета, как вы описали, чтобы проверить гипотезы."""
    df = clean_base(df_raw)

    # 01 полный
    v1 = df.copy()

    # 02 с BMI (оставим BMI и удалим Height/Weight, чтобы проверить гипотезу)
    v2 = add_bmi(df)
    v2 = v2.drop(columns=["Height", "Weight"])

    # 03 без роста
    v3 = df.drop(columns=["Height"])

    # 04 без веса
    v4 = df.drop(columns=["Weight"])

    # 05 без роста и веса
    v5 = df.drop(columns=["Height", "Weight"])

    # 06 только рост и вес (и таргет)
    keep = ["Height", "Weight", TARGET_COL]
    v6 = df[keep].copy()

    return {
        DatasetVariants.full_with_height_weight: v1,
        DatasetVariants.with_bmi: v2,
        DatasetVariants.no_height: v3,
        DatasetVariants.no_weight: v4,
        DatasetVariants.no_height_no_weight: v5,
        DatasetVariants.only_height_weight: v6,
    }


def save_variants(df_raw: pd.DataFrame, processed_dir: str | Path) -> List[Path]:
    """Сохраняет варианты в data/processed/ и возвращает список путей."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tables = make_variant_tables(df_raw)
    out_paths: List[Path] = []
    for fname, table in tables.items():
        p = processed_dir / fname
        table.to_csv(p, index=False)
        out_paths.append(p)
    return out_paths


def build_pipeline_for_training(
    df_variant: pd.DataFrame,
    *,
    model,
    onehot_nominal: bool = True,
    scale_numeric: bool = True,
) -> Pipeline:
    X, _ = split_xy(df_variant)
    preprocessor = build_preprocessor(X, onehot_nominal=onehot_nominal, scale_numeric=scale_numeric)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
