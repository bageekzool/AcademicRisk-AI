
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset location
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = RAW_DIR / "data.csv"


def load_dataset():
    """Download UCI dataset 697 or load its local copy."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists():
        print("[INFO] Loading dataset from local file...")
        df = pd.read_csv(DATASET_PATH)

    else:
        print("[INFO] Downloading dataset from UCI...")

        dataset = fetch_ucirepo(id=697)

        X = dataset.data.features
        y = dataset.data.targets

        df = pd.concat([X, y], axis=1)

        df.to_csv(DATASET_PATH, index=False)

        print("[SUCCESS] Dataset downloaded and saved.")

    # Basic validation
    expected_columns = 37

    if df.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, "
            f"but found {df.shape[1]}"
        )

    if "Target" not in df.columns:
        raise ValueError("Target column is missing.")

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\n===== DATASET INFORMATION =====")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nTarget Distribution:")
    print(df["Target"].value_counts())

    print("\nFirst 5 Rows:")
    print(df.head().to_string())
