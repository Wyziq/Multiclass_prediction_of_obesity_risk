from __future__ import annotations

from pathlib import Path
from src.data_preprocessing import load_data, export_prepared_data

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "ObesityDataSet.csv"
    nb_prepared = project_root / "notebooks" / "prepared_data.csv"

    df = load_data(raw_path)
    export_prepared_data(df, nb_prepared)
    print(f"Saved legacy prepared dataset to: {nb_prepared}")

if __name__ == "__main__":
    main()
