
from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"


def classify_stage(column):
    if column == "Target":
        return "TARGET"

    if column.startswith("Curricular units 2nd sem"):
        return "EXCLUDE_EARLY"

    if column.startswith("Curricular units 1st sem"):
        return "FIRST_SEMESTER"

    return "REVIEW_ENROLLMENT"


def audit_features():
    df = load_dataset()
    records = []

    for column in df.columns:
        values = df[column].dropna().unique()

        records.append({
            "feature": column,
            "dtype": str(df[column].dtype),
            "unique_count": df[column].nunique(),
            "missing_count": int(df[column].isna().sum()),
            "sample_values": str(values[:8].tolist()),
            "provisional_stage": classify_stage(column),
        })

    report = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "feature_audit.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== FEATURE AUDIT =====")
    print(report.to_string(index=False))

    print("\n===== PROVISIONAL STAGES =====")
    print(report["provisional_stage"].value_counts())

    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    audit_features()
