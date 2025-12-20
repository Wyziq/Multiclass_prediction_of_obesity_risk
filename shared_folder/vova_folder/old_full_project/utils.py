"""Utility helpers shared across notebooks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml


def project_root() -> Path:
    """Return absolute path to the project root (current file directory)."""
    return Path(__file__).resolve().parent


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
]
