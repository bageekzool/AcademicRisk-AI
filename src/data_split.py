
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset


# =========================================
# PROJECT CONFIGURATION
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TARGET = "Target"
RANDOM_STATE = 42

EXPECTED_CLASSES = {
    "Dropout",
    "Enrolled",
    "Graduate",
}


def create_data_splits():

    print("\n====================================")
    print("ACADEMICRISK AI - DATA SPLITTING")
    print("====================================")

    # 1. Load original dataset
    df = load_dataset()

    # 2. Validate dataset
    if TARGET not in df.columns:
        raise ValueError("Target column is missing.")

    actual_classes = set(df[TARGET].unique())

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected target classes: {actual_classes}"
        )

    if not df.index.is_unique:
        raise ValueError("Dataset index is not unique.")

    # 3. Split into 70% train and 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    # 4. Split temporary set equally:
    # 15% validation and 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET],
    )

    # 5. Validate split integrity
    train_ids = set(train_df.index)
    val_ids = set(val_df.index)
    test_ids = set(test_df.index)

    if train_ids & val_ids:
        raise ValueError("Train/validation overlap.")

    if train_ids & test_ids:
        raise ValueError("Train/test overlap.")

    if val_ids & test_ids:
        raise ValueError("Validation/test overlap.")

    all_ids = train_ids | val_ids | test_ids

    if len(all_ids) != len(df):
        raise ValueError(
            "Some dataset rows are missing from splits."
        )

    # 6. Preserve original row IDs
    train_df = train_df.sort_index()
    val_df = val_df.sort_index()
    test_df = test_df.sort_index()

    # 7. Create output directory
    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 8. Save all splits
    train_df.to_csv(
        SPLITS_DIR / "train.csv",
        index_label="row_id",
    )

    val_df.to_csv(
        SPLITS_DIR / "validation.csv",
        index_label="row_id",
    )

    test_df.to_csv(
        SPLITS_DIR / "test.csv",
        index_label="row_id",
    )

    # 9. Display summary
    print("\n========== SPLIT SUMMARY ==========")

    summary = []

    for name, subset in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        summary.append({
            "Split": name,
            "Students": len(subset),
            "Percentage": round(
                len(subset) / len(df) * 100,
                2
            ),
        })

    print(
        pd.DataFrame(summary).to_string(index=False)
    )

    # 10. Display target distribution
    print("\n========== CLASS DISTRIBUTION ==========")

    for name, subset in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print(f"\n{name}:")

        print(
            subset[TARGET]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_string()
        )

    print("\n========== VALIDATION ==========")
    print("[PASS] No overlapping rows")
    print("[PASS] All dataset rows accounted for")
    print("[PASS] Stratified split completed")
    print("[PASS] Original row IDs preserved")

    print("\n[SUCCESS] Data splitting completed.")


if __name__ == "__main__":
    create_data_splits()
