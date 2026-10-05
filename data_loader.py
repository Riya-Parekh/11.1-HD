from pathlib import Path
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv"
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def load_data(data_path: str = "data/heart.csv") -> pd.DataFrame:
    """Load the 1,025-row heart dataset, downloading it if not present locally."""
    path = Path(data_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        print("Local data/heart.csv not found; downloading the documented source dataset...")
        df = pd.read_csv(DATA_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema. Expected columns: " + ", ".join(EXPECTED_COLUMNS)
        )
    return df


def create_deduplicated_data(df: pd.DataFrame, output_path: str = "data/heart_deduplicated.csv") -> pd.DataFrame:
    """Remove exact duplicate rows and save the 302-row unique dataset."""
    dedup = df.drop_duplicates().reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dedup.to_csv(output_path, index=False)
    return dedup


if __name__ == "__main__":
    df = load_data()
    clean = create_deduplicated_data(df)
    print(f"Original shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Deduplicated shape: {clean.shape}")
