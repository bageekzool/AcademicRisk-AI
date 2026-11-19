
from pathlib import Path

import yaml

from src.data_loader import load_dataset


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "feature_sets.yaml"


def validate_feature_sets():

    print("\n====================================")
    print("ACADEMICRISK AI - FEATURE VALIDATION")
    print("====================================")

    # 1. Load configuration
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    # 2. Load actual dataset
    df = load_dataset()

    target = config["target"]

    features_a = config["enrollment_features"]
    features_first_sem = config["first_semester_features"]
    features_second_sem = config["second_semester_features"]

    # 3. Validate lists
    feature_groups = {
        "enrollment_features": features_a,
        "first_semester_features": features_first_sem,
        "second_semester_features": features_second_sem,
    }

    for group_name, features in feature_groups.items():

        if not isinstance(features, list):
            raise ValueError(
                f"{group_name} must be a list."
            )

        if not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError(
                f"{group_name} contains invalid names."
            )

        if len(features) != len(set(features)):
            raise ValueError(
                f"Duplicate features in {group_name}."
            )

    # 4. Build Model A and Model B
    features_b = features_a + features_first_sem

    # 5. Check overlap between groups
    for group1, group2 in [
        ("enrollment_features", "first_semester_features"),
        ("enrollment_features", "second_semester_features"),
        ("first_semester_features", "second_semester_features"),
    ]:
        overlap = set(feature_groups[group1]) & set(
            feature_groups[group2]
        )

        if overlap:
            raise ValueError(
                f"Feature overlap: {group1} / {group2}: "
                f"{sorted(overlap)}"
            )

    # 6. Check column existence
    all_configured = (
        features_a
        + features_first_sem
        + features_second_sem
    )

    missing_columns = [
        feature for feature in all_configured
        if feature not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    if target not in df.columns:
        raise ValueError("Target column not found.")

    if target in all_configured:
        raise ValueError(
            "DATA LEAKAGE: Target included as input!"
        )

    # 7. Verify academic timeline
    if any(
        feature.startswith("Curricular units")
        for feature in features_a
    ):
        raise ValueError(
            "Model A includes semester features!"
        )

    if any(
        feature.startswith("Curricular units 2nd sem")
        for feature in features_b
    ):
        raise ValueError(
            "Model B includes second-semester features!"
        )

    # 8. Build X and y for validation
    X_a = df[features_a]
    X_b = df[features_b]
    y = df[target]

    if not (
        len(X_a) == len(X_b) == len(y)
    ):
        raise ValueError(
            "Input and target row counts differ."
        )

    # 9. Print report
    print("\n========== MODEL A ==========")
    print("Stage: Enrollment")
    print("Students:", X_a.shape[0])
    print("Features:", X_a.shape[1])

    print("\n========== MODEL B ==========")
    print("Stage: First Semester")
    print("Students:", X_b.shape[0])
    print("Features:", X_b.shape[1])

    print("\n========== TARGET ==========")
    print("Target:", target)
    print("Classes:", sorted(y.unique().tolist()))

    print("\n========== VALIDATION ==========")
    print("[PASS] All features exist in the dataset")
    print("[PASS] No duplicate or overlapping features")
    print("[PASS] Target excluded from model inputs")
    print("[PASS] Model A excludes semester features")
    print("[PASS] Model B excludes second-semester features")
    print("[PASS] Input and target row counts match")

    print("\n[SUCCESS] Feature sets validated!")


if __name__ == "__main__":
    validate_feature_sets()
