
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.data_loader import load_dataset


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

DICTIONARY_PATH = OUTPUT_DIR / "feature_dictionary.csv"


# ============================================================
# 2. PROVISIONAL FEATURE CLASSIFICATION
# ============================================================

def classify_stage(column: str) -> str:
    """
    Classify each feature according to its possible
    availability during the academic timeline.

    This classification is provisional.

    REVIEW_ENROLLMENT:
        Needs manual verification before being used
        at enrollment time.

    FIRST_SEMESTER:
        First-semester academic information.

    EXCLUDE_EARLY:
        Second-semester information that must not
        be used in early prediction models.

    TARGET:
        Prediction target, never an input feature.
    """

    if column == "Target":
        return "TARGET"

    if column.startswith("Curricular units 2nd sem"):
        return "EXCLUDE_EARLY"

    if column.startswith("Curricular units 1st sem"):
        return "FIRST_SEMESTER"

    return "REVIEW_ENROLLMENT"


# ============================================================
# 3. LOAD OFFICIAL METADATA
# ============================================================

def load_official_metadata() -> pd.DataFrame:
    """
    Fetch official feature metadata from UCI dataset 697.
    """

    print("[INFO] Fetching official UCI metadata...")

    dataset = fetch_ucirepo(id=697)

    variables = dataset.variables.copy()

    if variables is None or variables.empty:
        raise ValueError(
            "UCI returned an empty variable dictionary."
        )

    required_columns = [
        "name",
        "role",
        "type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in variables.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required metadata columns: "
            f"{missing_columns}"
        )

    if variables["name"].isna().any():
        raise ValueError(
            "UCI metadata contains empty feature names."
        )

    if variables["name"].duplicated().any():
        duplicates = variables.loc[
            variables["name"].duplicated(),
            "name"
        ].tolist()

        raise ValueError(
            f"Duplicate feature names found: {duplicates}"
        )

    return variables


# ============================================================
# 4. VALIDATE DATASET AND METADATA
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    variables: pd.DataFrame,
) -> None:
    """
    Ensure local dataset columns match official UCI metadata.
    """

    local_columns = set(df.columns)

    official_columns = set(variables["name"])

    missing_in_metadata = (
        local_columns - official_columns
    )

    extra_in_metadata = (
        official_columns - local_columns
    )

    if missing_in_metadata or extra_in_metadata:
        raise ValueError(
            "Dataset and UCI metadata do not match.\n"
            f"Missing in metadata: {missing_in_metadata}\n"
            f"Extra in metadata: {extra_in_metadata}"
        )

    if len(df.columns) != 37:
        raise ValueError(
            f"Expected 37 columns, found {len(df.columns)}."
        )

    print("[SUCCESS] Dataset and metadata columns match.")


# ============================================================
# 5. BUILD FEATURE DICTIONARY
# ============================================================

def build_feature_dictionary() -> pd.DataFrame:
    """
    Generate a feature dictionary containing:

    - Official feature names
    - Official feature roles and types
    - Official descriptions, when available
    - Actual dataset data types
    - Unique value counts
    - Missing value counts
    - Provisional academic-stage classification
    - Manual review fields

    Save the resulting table as CSV.
    """

    print("\n====================================")
    print("ACADEMICRISK AI - FEATURE DICTIONARY")
    print("====================================\n")

    # --------------------------------------------------------
    # STEP 1: LOAD LOCAL DATASET
    # --------------------------------------------------------

    print("[INFO] Loading local dataset...")

    df = load_dataset()

    # --------------------------------------------------------
    # STEP 2: LOAD UCI METADATA
    # --------------------------------------------------------

    variables = load_official_metadata()

    # --------------------------------------------------------
    # STEP 3: VALIDATE COLUMNS
    # --------------------------------------------------------

    validate_columns(df, variables)

    # --------------------------------------------------------
    # STEP 4: ENSURE OPTIONAL METADATA FIELDS EXIST
    # --------------------------------------------------------

    optional_columns = [
        "description",
        "demographic",
        "units",
        "missing_values",
    ]

    for column in optional_columns:
        if column not in variables.columns:
            variables[column] = pd.NA

    # --------------------------------------------------------
    # STEP 5: PRESERVE LOCAL DATASET COLUMN ORDER
    # --------------------------------------------------------

    # Explicitly rename the index after reindexing.
    # This prevents the previous KeyError: ['name'].

    variables = (
        variables
        .set_index("name")
        .reindex(df.columns)
        .rename_axis("name")
        .reset_index()
    )

    if "name" not in variables.columns:
        raise RuntimeError(
            "Feature name column was not restored."
        )

    # --------------------------------------------------------
    # STEP 6: ADD ACTUAL DATASET STATISTICS
    # --------------------------------------------------------

    dtype_map = {
        column: str(df[column].dtype)
        for column in df.columns
    }

    unique_map = {
        column: int(df[column].nunique())
        for column in df.columns
    }

    missing_map = {
        column: int(df[column].isna().sum())
        for column in df.columns
    }

    variables["actual_dtype"] = (
        variables["name"].map(dtype_map)
    )

    variables["unique_count"] = (
        variables["name"].map(unique_map)
    )

    variables["actual_missing_count"] = (
        variables["name"].map(missing_map)
    )

    # --------------------------------------------------------
    # STEP 7: CLASSIFY ACADEMIC STAGE
    # --------------------------------------------------------

    variables["provisional_stage"] = (
        variables["name"].map(classify_stage)
    )

    # Final decisions are not automatic.
    variables["final_decision"] = "PENDING_REVIEW"

    variables["review_notes"] = ""

    # --------------------------------------------------------
    # STEP 8: ARRANGE OUTPUT COLUMNS
    # --------------------------------------------------------

    output_columns = [
        "name",
        "role",
        "type",
        "description",
        "demographic",
        "units",
        "missing_values",
        "actual_dtype",
        "unique_count",
        "actual_missing_count",
        "provisional_stage",
        "final_decision",
        "review_notes",
    ]

    variables = variables[output_columns].copy()

    # --------------------------------------------------------
    # STEP 9: SAVE FEATURE DICTIONARY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    variables.to_csv(
        DICTIONARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SUCCESS] Feature dictionary saved.")

    # --------------------------------------------------------
    # STEP 10: DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n========== DATA DICTIONARY ==========")

    display_columns = [
        "name",
        "role",
        "type",
        "provisional_stage",
        "final_decision",
    ]

    print(
        variables[display_columns].to_string(
            index=False
        )
    )

    print("\n========== SUMMARY ==========")

    print(f"Total variables: {len(variables)}")

    print("\nStage distribution:")

    print(
        variables["provisional_stage"].value_counts()
    )

    print("\n========== DATA QUALITY ==========")

    print(
        "Total actual missing values:",
        int(variables["actual_missing_count"].sum())
    )

    print(
        "Features pending manual review:",
        int(
            (
                variables["final_decision"]
                == "PENDING_REVIEW"
            ).sum()
        )
    )

    print("\n[SAVED FILE]")
    print(DICTIONARY_PATH)

    return variables


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_feature_dictionary()
