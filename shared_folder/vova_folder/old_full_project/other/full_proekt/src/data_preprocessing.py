
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



def export_prepared_data(
    df: pd.DataFrame,
    output_path: str | Path,
    include_target: bool = True,
) -> Path:
    """Экспортирует полностью числовой датасет (для legacy-ноутбуков svm/logreg).

    Особенность: legacy-ноутбуки ожидают файл 'prepared_data.csv' с уже закодированными
    категориальными признаками. Поэтому здесь:
      - применяем clean_base (округление дискретных + бинарные yes/no + ординальные mapping)
      - one-hot кодируем номинальные (Gender, MTRANS) через pandas.get_dummies
      - target NObeyesdad оставляем строковым (как в исходном датасете)
    """
    out = clean_base(df)

    # one-hot для номинальных
    nominal_cols = [c for c in NOMINAL_CATEGORICAL if c in out.columns]
    if nominal_cols:
        out = pd.get_dummies(out, columns=nominal_cols, drop_first=False)

    if not include_target and TARGET_COL in out.columns:
        out = out.drop(columns=[TARGET_COL])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path

# ================================
# Helpers for unified training/predict
# ================================
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TARGET_COL = "NObeyesdad"

DISCRETE_NUM_COLS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
YES_NO_COLS = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ORDINAL_MAPS = {
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
}

def load_raw_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _base_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DISCRETE_NUM_COLS:
        if c in df.columns:
            df[c] = df[c].round().astype(int)
    for c in YES_NO_COLS:
        if c in df.columns:
            df[c] = df[c].map({"yes": 1, "no": 0}).astype(int)
    for c, mp in ORDINAL_MAPS.items():
        if c in df.columns:
            df[c] = df[c].map(mp).astype(int)
    return df

def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Weight" in df.columns and "Height" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df

def save_dataset_variants(df: pd.DataFrame, out_dir: Path) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df0 = _base_clean(df)

    variants: Dict[str, pd.DataFrame] = {}

    def _save(name: str, d: pd.DataFrame):
        variants[name] = d
        (out_dir / f"{name}.csv").write_text(d.to_csv(index=False), encoding="utf-8")

    _save("01_full_with_height_weight", df0)
    _save("02_with_bmi", add_bmi(df0))
    _save("03_no_height", df0.drop(columns=["Height"]) if "Height" in df0.columns else df0)
    _save("04_no_weight", df0.drop(columns=["Weight"]) if "Weight" in df0.columns else df0)

    drop_hw = [c for c in ["Height", "Weight"] if c in df0.columns]
    _save("05_no_height_no_weight", df0.drop(columns=drop_hw))

    keep = [c for c in ["Height", "Weight", TARGET_COL] if c in df0.columns]
    _save("06_only_height_weight", df0[keep])

    return variants

def make_feature_target(df: pd.DataFrame, allow_missing_target: bool = False):
    df = df.copy()
    if TARGET_COL in df.columns:
        return df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    if allow_missing_target:
        return df, None
    raise KeyError(f"Target column '{TARGET_COL}' not found")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )
