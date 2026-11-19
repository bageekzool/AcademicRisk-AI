
import pandas as pd

from src.data_loader import load_dataset


def inspect_dataset():
    df = load_dataset()

    print("\n========== DATASET SHAPE ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n========== COLUMN TYPES ==========")
    for i, (column, dtype) in enumerate(
        df.dtypes.items(), start=1
    ):
        print(f"{i:02d}. {column} --> {dtype}")

    print("\n========== MISSING VALUES ==========")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print("Total missing:", missing.sum())

    print("\n========== DUPLICATE ROWS ==========")
    print("Duplicates:", df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["Target"].value_counts())

    print("\n========== TARGET PERCENTAGES ==========")
    print(
        df["Target"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n========== NUMERIC SUMMARY ==========")
    print(df.describe().to_string())

    print("\n========== MEMORY USAGE ==========")
    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )
    print(f"Memory usage: {memory_mb:.2f} MB")


if __name__ == "__main__":
    inspect_dataset()
