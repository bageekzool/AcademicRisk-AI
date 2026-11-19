
from pathlib import Path

import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "feature_sets.yaml"

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


# ==========================================
# LOAD CONFIGURATION
# ==========================================

def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================
# GET FEATURES FOR MODEL A OR B
# ==========================================

def get_features(config, stage):

    enrollment = config["enrollment_features"]

    if stage == "A":
        return enrollment.copy()

    if stage == "B":
        return (
            enrollment
            + config["first_semester_features"]
        )

    raise ValueError("Stage must be A or B.")


# ==========================================
# BUILD PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor(config, features, model_type):

    categorical = [
        f for f in config["categorical_features"]
        if f in features
    ]

    binary = [
        f for f in config["binary_features"]
        if f in features
    ]

    numeric = [
        f for f in features
        if f not in categorical + binary
    ]

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        ),
    ])

    if model_type == "linear":

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ])

    elif model_type == "tree":

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ])

    else:
        raise ValueError(
            "model_type must be 'linear' or 'tree'."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "binary",
                "passthrough",
                binary
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ==========================================
# TEST PREPROCESSING
# ==========================================

def test_preprocessing():

    print("\n====================================")
    print("ACADEMICRISK AI - PREPROCESSING TEST")
    print("====================================")

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    target = config["target"]

    for stage in ["A", "B"]:

        features = get_features(config, stage)

        X_train = train_df[features]
        X_val = val_df[features]

        y_train = train_df[target]
        y_val = val_df[target]

        print(f"\n========== MODEL {stage} ==========")

        print("Original features:", len(features))

        for model_type in ["linear", "tree"]:

            preprocessor = build_preprocessor(
                config,
                features,
                model_type
            )

            # Fit ONLY on training data
            X_train_processed = (
                preprocessor.fit_transform(X_train)
            )

            # Validation uses the SAME fitted transformer
            X_val_processed = (
                preprocessor.transform(X_val)
            )

            print(f"\nPreprocessor: {model_type}")

            print(
                "Training shape:",
                X_train_processed.shape
            )

            print(
                "Validation shape:",
                X_val_processed.shape
            )

            if (
                X_train_processed.shape[1]
                != X_val_processed.shape[1]
            ):
                raise ValueError(
                    "Feature count mismatch!"
                )

            print("[PASS] Feature dimensions match")

        print("Training targets:", len(y_train))
        print("Validation targets:", len(y_val))

    print("\n[SUCCESS] Preprocessing test completed.")


if __name__ == "__main__":
    test_preprocessing()
