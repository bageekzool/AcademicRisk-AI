
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)


ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"


def main():

    print("\n====================================")
    print("ACADEMICRISK AI - FREEZE CANDIDATES")
    print("====================================")

    config = load_config()

    # Training data only
    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    features = get_features(config, "B")

    X_train = train_df[features]
    y_train = train_df[config["target"]]

    print("\n[STEP 1] Training Multiclass B...")
    print("Algorithm: Logistic Regression")
    print("Class Weight: balanced")
    print("C: 1.0")
    print("Features:", len(features))
    print("Training students:", len(train_df))

    pipeline = Pipeline([
        (
            "preprocessing",
            build_preprocessor(
                config,
                features,
                "linear",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR /
        "multiclass_logistic_B_balanced.joblib"
    )

    joblib.dump(pipeline, model_path)

    print("[SAVED MODEL]", model_path)

    # ======================================
    # VERIFY EXISTING CANDIDATE ARTIFACTS
    # ======================================

    print("\n[STEP 2] Verifying candidate models...")

    candidates = {
        "multiclass_A": {
            "file": "random_forest_A.joblib",
            "stage": "A",
            "type": "multiclass",
            "selection_metric": "macro_f1",
            "validation_score": 0.4745,
        },
        "multiclass_B": {
            "file": "multiclass_logistic_B_balanced.joblib",
            "stage": "B",
            "type": "multiclass",
            "selection_metric": "macro_f1",
            "validation_score": 0.6346,
        },
        "binary_warning_B": {
            "file": "binary_logistic_B_threshold.joblib",
            "stage": "B",
            "type": "binary",
            "selection_metric": "f2",
            "validation_score": 0.7561,
            "threshold": 0.22,
        },
    }

    for name, metadata in candidates.items():

        path = MODELS_DIR / metadata["file"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing candidate: {path}"
            )

        artifact = joblib.load(path)

        if name == "binary_warning_B":

            saved_threshold = artifact["threshold"]

            if abs(saved_threshold - 0.22) > 1e-9:
                raise ValueError(
                    "Binary threshold mismatch!"
                )

        print(f"[VERIFIED] {name}: {metadata['file']}")

    # ======================================
    # SAVE CANDIDATE MANIFEST
    # ======================================

    manifest = {
        "project": "AcademicRisk AI",
        "status": "frozen_candidates_before_test",
        "data_split": {
            "training": 3096,
            "validation": 664,
            "test": 664,
        },
        "notes": [
            "Candidate scores are from Validation.",
            "Test data has not been used.",
            "Binary threshold was selected using "
            "training out-of-fold predictions.",
            "Final test scores are not yet available.",
        ],
        "candidates": candidates,
    }

    manifest_path = (
        CONFIG_DIR /
        "selected_models.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n[SAVED MANIFEST]", manifest_path)

    print("\n====================================")
    print("CANDIDATE SELECTION COMPLETED")
    print("====================================")

    print("[SUCCESS] Models frozen before test.")


if __name__ == "__main__":
    main()
