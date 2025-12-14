
from __future__ import annotations

from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import pandas as pd
import yaml

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib

# ------ Utility functions by Polina ------

def project_root() -> Path:
    """Return absolute path to the project root (one level above `src`)."""
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Read a YAML file located relative to the project root."""
    target = project_root() / Path(path)
    with target.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


DATA_DIR = project_root()


def get_data_path(*relative: str) -> Path:
    """
    Resolve paths relative to the project directory.

    Usage:
        raw = get_data_path("dataset.csv")
    """
    return DATA_DIR.joinpath(*relative)


def load_columns_mapping() -> Dict[str, Any]:
    """
    Load and return the project mapping from `columns_mapping.yml`.

    Mapping schema:
      - `globals`: project-wide settings (e.g. `groups_order`)
      - `columns`: metadata for feature columns
      - `targets`: metadata for target columns
    """
    return _load_columns_mapping_uncached()


COLUMNS_MAPPING_FILE = "columns_mapping.yml"


@lru_cache(maxsize=1)
def _load_columns_mapping_uncached() -> Dict[str, Any]:
    return load_yaml(COLUMNS_MAPPING_FILE)


def clear_columns_mapping_cache() -> None:
    """Clear mapping cache (useful when editing YAML during a notebook session)."""
    _load_columns_mapping_uncached.cache_clear()

def cm_globals() -> Dict[str, Any]:
    """Return the `globals` section of mapping."""
    mapping = load_columns_mapping()
    section = mapping.get("globals", {})
    return section if isinstance(section, dict) else {}


def cm_columns() -> Dict[str, Dict[str, Any]]:
    """Return the `columns` section of mapping."""
    mapping = load_columns_mapping()
    section = mapping.get("columns", {})
    return section if isinstance(section, dict) else {}


def cm_targets() -> Dict[str, Dict[str, Any]]:
    """Return the `targets` section of mapping."""
    mapping = load_columns_mapping()
    section = mapping.get("targets", {})
    return section if isinstance(section, dict) else {}


def cm_groups_order() -> list[str]:
    """Return global group order used for sorting."""
    value = cm_globals().get("groups_order", [])
    return value if isinstance(value, list) else []


def cm_group_rank() -> Dict[str, int]:
    """Return `{group_name: rank}` for sorting by groups."""
    return {g: i for i, g in enumerate(cm_groups_order())}


def cm_field_meta(name: str) -> Dict[str, Any]:
    """
    Return metadata for a field (feature or target).

    Looks up `name` in both `columns` and `targets`.
    """
    cols = cm_columns()
    meta = cols.get(name)
    if isinstance(meta, dict):
        return meta
    targets = cm_targets()
    meta = targets.get(name)
    return meta if isinstance(meta, dict) else {}


def cm_label(
    name: str,
    key: str = "description_ru",
    default: Optional[str] = None,
) -> str:
    """Return a human-readable label for a column/target (default: russian description)."""
    meta = cm_field_meta(name)
    value = meta.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return name if default is None else default


def cm_group(
    name: str,
    default: str = "Прочие признаки",
) -> str:
    """Return group name (`group_ru`) for a field."""
    meta = cm_field_meta(name)
    value = meta.get("group_ru")
    return value if isinstance(value, str) and value.strip() else default


def cm_order(name: str) -> Optional[list[str]]:
    """Return ordering list (`order`) for an ordinal field, if present."""
    meta = cm_field_meta(name)
    value = meta.get("order")
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


def cm_target_mapping(target: str) -> Dict[str, str]:
    """Return class mapping dict for a target (e.g. NObeyesdad_norm.mapping)."""
    meta = cm_targets().get(target, {})
    value = meta.get("mapping", {})
    return value if isinstance(value, dict) else {}


def cm_target_categories(target: str) -> Dict[str, Dict[str, Any]]:
    """Return categories dict for a target (e.g. NObeyesdad_norm.categories)."""
    meta = cm_targets().get(target, {})
    value = meta.get("categories", {})
    return value if isinstance(value, dict) else {}


def cm_target_order(target: str) -> Optional[list[str]]:
    """Return class order list for a target, if present."""
    meta = cm_targets().get(target, {})
    value = meta.get("order")
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


def cm_target_category_label(
    target: str,
    category: str,
    key: str = "description_ru",
    default: Optional[str] = None,
) -> str:
    """Return label for a specific target class (e.g. description_ru for 'Obesity')."""
    categories = cm_target_categories(target)
    meta = categories.get(category, {})
    if isinstance(meta, dict):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return category if default is None else default


def cm_labels_dict(
    names: Iterable[str],
    key: str = "description_ru",
) -> Dict[str, str]:
    """Return `{name: label}` for a list of column/target names."""
    return {name: cm_label(name, key=key) for name in names}


DEFAULT_RAW_DATASET = "ObesityDataSet.csv"


def load_csv(
    name: str = DEFAULT_RAW_DATASET,
    subdir: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file located in the project folder (optionally within a subfolder).

    Args:
        name: File name, defaults to the main dataset.
        subdir: Optional subfolder within the project root.
        kwargs: Extra keyword arguments forwarded to ``pandas.read_csv``.
    """
    path = get_data_path(subdir, name)
    if not path.exists() and not subdir:
        # Common project layouts: keep notebooks working even if data isn't in the root.
        candidates = [
            get_data_path("data", "raw", name),
            get_data_path("data", name),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            tried = [str(path), *map(str, candidates)]
            raise FileNotFoundError(f"CSV not found. Tried: {', '.join(tried)}")
    return pd.read_csv(path, **kwargs)


def _rename_columns(df: pd.DataFrame, column_names: Optional[str]) -> pd.DataFrame:
    """
    Rename columns using mapping from YAML.

    If ``column_names`` is None — no переименование.
    If строка — берётся значение по этому ключу из маппинга (например,
    ``short_ru``, ``description_ru``, ``description_en`` или любой новый
    ключ, который появится в YAML).
    """
    if column_names is None:
        return df

    # Aliases used in notebooks.
    # Keeping them here allows older notebooks to work without edits.
    scheme = {
        "short_names": "short_ru",
        "long_names": "description_ru",
    }.get(column_names, column_names)

    mapping = load_columns_mapping()
    rename_map = {}
    for col, meta in {**cm_columns(), **cm_targets()}.items():
        if isinstance(meta, dict):
            rename_map[col] = meta.get(scheme, col)
    # Если в df есть колонки, отсутствующие в маппинге, оставляем их как есть.
    rename_map.update({col: col for col in df.columns if col not in rename_map})
    return df.rename(columns=rename_map)


def load_raw_df(column_names: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw obesity dataset stored directly in the project directory.

    Args:
        column_names: optional renaming scheme: "short_names", "long_names", or None.
    """
    df = load_csv(name=DEFAULT_RAW_DATASET, subdir="")
    return _rename_columns(df, column_names)


def load_clean_df(
    column_names: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and clean the raw dataset by removing duplicate rows and
    resetting the index. Adds укрупнённую целевую переменную
    ``NObeyesdad_norm`` согласно маппингу в columns_mapping.yml.
    Returns a fresh DataFrame.

    Args:
        column_names: optional renaming scheme: "short_names", "long_names", or None.
    """
    # Всегда работаем с исходными именами колонок, затем переименовываем в конце.
    df = load_raw_df(column_names=None)
    df = df.drop_duplicates().reset_index(drop=True)

    # Добавляем агрегированную целевую переменную, если есть маппинг.
    mapping = load_columns_mapping()
    norm_mapping = cm_target_mapping("NObeyesdad_norm")
    if "NObeyesdad" in df.columns and norm_mapping:
        df["NObeyesdad_norm"] = df["NObeyesdad"].map(norm_mapping)

    # Добавляем индекс массы тела (BMI), если доступны вес и рост.
    if {"Weight", "Height"}.issubset(df.columns):
        height_sq = pd.to_numeric(df["Height"], errors="coerce") ** 2
        weight = pd.to_numeric(df["Weight"], errors="coerce")
        df["BMI"] = weight / height_sq
        df["BMI"] = df["BMI"].replace([float("inf"), -float("inf")], pd.NA)

    # --- Доп. обработка категориальных признаков ---
    for col in ("NObeyesdad_norm", "NObeyesdad", "Gender", "CAEC", "CALC", "FAVC", "MTRANS", "SMOKE", "family_history_with_overweight", "SCC"):
        df[col] = df[col].astype("category")

    # корректируем CH2O
    df_int = pd.to_numeric(df["CH2O"], errors="coerce").round().astype("Int64")
    df_map = {1: "<1", 2: "1-2", 3: ">2"}
    df["CH2O"] = df_int.map(df_map)
    df["CH2O"] = df["CH2O"].astype("category")

    # корректируем FCVC
    df_int = pd.to_numeric(df["FCVC"], errors="coerce").round().astype("Int64")
    df_map = {1: "no", 2: "Sometimes", 3: "Always"}
    df["FCVC"] = df_int.map(df_map)
    df["FCVC"] = df["FCVC"].astype("category")

    # корректируем NCP
    df_int = pd.to_numeric(df["NCP"], errors="coerce").round().astype("Int64")
    df_map = {1: "1-2", 2: "1-2", 3: "3", 4: ">3"}
    df["NCP"] = df_int.map(df_map)
    df["NCP"] = df["NCP"].astype("category")

    # корректируем FAF
    df_int = pd.to_numeric(df["FAF"], errors="coerce").round().astype("Int64")
    df_map = {0: "0", 1: "1-2", 2: "2-4", 3: "4-5"}
    df["FAF"] = df_int.map(df_map)
    df["FAF"] = df["FAF"].astype("category")

    # корректируем TUE
    df_int = pd.to_numeric(df["TUE"], errors="coerce").round().astype("Int64")
    df_map = {0: "0-2", 1: "3-5", 2: ">5"}
    df["TUE"] = df_int.map(df_map)
    df["TUE"] = df["TUE"].astype("category")

    # Переименовываем колонки согласно маппингу, если указана схема.
    df = _rename_columns(df, column_names)

    return df


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    *,
    ax=None,
    bins: int = 20,
    numeric_color: str = "#4C72B0",
    categorical_color: str = "#55A868",
):
    """
    Plot distribution for a single feature.

    - Numeric features -> histogram.
    - Categorical features -> bar chart (uses YAML `order` if present).

    Args:
        df: Source dataframe.
        feature: Column name.
        ax: Optional matplotlib Axes (when plotting inside a grid). If None, creates its own figure.
        bins: Histogram bins for numeric features.
        numeric_color: Color for histogram.
        categorical_color: Color for bar chart.
    """
    # Local import: utils.py should stay lightweight for non-plot use-cases.
    import matplotlib.pyplot as plt  # type: ignore

    if feature not in df.columns:
        raise KeyError(f"Column not found: {feature}")

    title = cm_label(feature)
    series = df[feature]

    numeric = pd.to_numeric(series, errors="coerce")
    # Heuristic: treat as numeric if dtype is numeric OR most values are convertible.
    is_numeric = pd.api.types.is_numeric_dtype(series) or (
        numeric.notna().any() and numeric.notna().mean() > 0.8
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if is_numeric:
        values = numeric.dropna()
        ax.hist(values, bins=bins, color=numeric_color)
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel("Количество")
        return values

    ordering = cm_order(feature)
    counts = series.value_counts(dropna=False)
    if ordering:
        cat = pd.Categorical(series, categories=ordering, ordered=True)
        counts = pd.Series(cat).value_counts(sort=False)
        extra = series[~series.isin(ordering)]
        if not extra.empty:
            counts = pd.concat([counts, extra.value_counts()])

    order_labels = counts.index.astype(str).tolist()
    order_values = counts.values.tolist()
    positions = list(range(len(order_labels)))
    ax.bar(positions, order_values, color=categorical_color)
    ax.set_xticks(positions)
    ax.set_xticklabels(order_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Количество")
    return counts


# ------ Feature engineering helpers (from 05_Feature_Engineering, by Vova) ------


def make_feature_sets(
    df: pd.DataFrame,
    *,
    target_col: str = "NObeyesdad_norm",
    drop_cols: Optional[list[str]] = None,
    body_cols: Optional[list[str]] = None,
) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare feature-set variants for experiments.

    By default follows 05_Feature_Engineering.ipynb:
      - target: NObeyesdad_norm
      - drop from features: NObeyesdad, BMI
      - two variants: with and without anthropometry (Height/Weight)
    """
    if drop_cols is None:
        drop_cols = ["NObeyesdad", "BMI"]
    if body_cols is None:
        # Support both original and renamed columns.
        body_candidates = ["Height", "Weight", "Рост", "Вес"]
        body_cols = [c for c in body_candidates if c in df.columns]
        if not body_cols:
            body_cols = ["Height", "Weight"]

    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    y = df[target_col]
    X_all = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    X_with_body = X_all.copy()
    body_cols_present = [c for c in body_cols if c in X_all.columns]
    X_no_body = X_all.drop(columns=body_cols_present, errors="ignore")

    return {
        "with_body": (X_with_body, y),
        "no_body": (X_no_body, y),
    }


def make_features_info_df(
    X: pd.DataFrame,
    *,
    default_group: str = "(без группы)",
) -> pd.DataFrame:
    """Return features info table: group, ru label, dtype (sorted by mapping group order)."""
    group_rank = cm_group_rank()
    rows = []
    for col in X.columns:
        group = cm_group(col, default=default_group)
        rows.append(
            {
                "group_ru": group,
                "group_rank": group_rank.get(group, 999),
                "feature": col,
                "description_ru": cm_label(col),
                "dtype": str(X[col].dtype),
            }
        )

    info_df = pd.DataFrame(rows)
    if info_df.empty:
        return info_df

    return (
        info_df.sort_values(["group_rank", "group_ru", "feature"])
        .drop(columns=["group_rank"])
        .reset_index(drop=True)
    )


def build_preprocessor_vova(
    X: pd.DataFrame,
    *,
    onehot_nominal: bool = True,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer using mapping metadata:
    - numeric: median impute (+ scaling optionally)
    - categorical with `order` in columns_mapping.yml: OrdinalEncoder
    - other categorical: OneHotEncoder (by default)
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    ordinal_cols = [c for c in cat_cols if cm_order(c)]
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]

    transformers = []

    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), num_cols))

    if ordinal_cols:
        categories = [cm_order(c) for c in ordinal_cols]
        ord_encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        ord_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", ord_encoder),
            ]
        )
        transformers.append(("ord", ord_pipe, ordinal_cols))

    if nominal_cols:
        if onehot_nominal:
            nom_encoder = OneHotEncoder(handle_unknown="ignore")
            nom_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", nom_encoder),
                ]
            )
            transformers.append(("nom", nom_pipe, nominal_cols))
        else:
            transformers.append(("nom", "passthrough", nominal_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline_for_training_vova(
    X: pd.DataFrame,
    *,
    model,
    onehot_nominal: bool = True,
    scale_numeric: bool = True,
) -> Pipeline:
    preprocessor = build_preprocessor_vova(
        X, onehot_nominal=onehot_nominal, scale_numeric=scale_numeric
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


# ------ Utility functions by Polina ------


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def save_joblib(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def evaluate_cv(
    pipeline,
    X,
    y,
    *,
    cv_splits: int = 5,
    seed: int = 42,
    scoring: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Оценивает pipeline кросс-валидацией только на train (без утечек)."""
    if scoring is None:
        scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    summary = {k.replace("test_", ""): float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}
    summary["cv_splits"] = cv_splits
    summary["seed"] = seed
    return summary


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# -----------



__all__ = [
    "project_root",
    "load_yaml",
    "get_data_path",
    "load_columns_mapping",
    "clear_columns_mapping_cache",
    "cm_globals",
    "cm_columns",
    "cm_targets",
    "cm_groups_order",
    "cm_group_rank",
    "cm_field_meta",
    "cm_label",
    "cm_group",
    "cm_order",
    "cm_target_mapping",
    "cm_target_categories",
    "cm_target_order",
    "cm_target_category_label",
    "cm_labels_dict",
    "load_csv",
    "DATA_DIR",
    "DEFAULT_RAW_DATASET",
    "load_raw_df",
    "load_clean_df",
    "plot_feature_distribution",
    "make_feature_sets",
    "make_features_info_df",
    "build_preprocessor_vova",
    "build_pipeline_for_training_vova",
    "set_seed",
    "save_joblib",
    "load_joblib",
    "evaluate_cv",
    "save_metrics",
]
