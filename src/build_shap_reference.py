
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from scipy.special import expit


# ============================================
# PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    ROOT / "data" / "splits" / "train.csv"
)

MODEL_PATH = (
    ROOT / "models"
    / "binary_logistic_B_threshold.joblib"
)

OUTPUT_PATH = (
    ROOT / "models"
    / "binary_B_shap_reference.joblib"
)

RANDOM_STATE = 42
BACKGROUND_SIZE = 150


# ============================================
# HELPERS
# ============================================

def to_dense(matrix):

    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


# ============================================
# MAIN
# ============================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - SHAP REFERENCE")
    print("====================================")

    # ----------------------------------------
    # STEP 1: Load frozen binary model
    # ----------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError(
            "Expected a saved model dictionary."
        )

    pipeline = artifact["pipeline"]
    features = list(artifact["features"])
    threshold = float(artifact["threshold"])

    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    if list(classifier.classes_) != [0, 1]:
        raise ValueError(
            "Unexpected binary class order."
        )

    if classifier.coef_.shape[0] != 1:
        raise ValueError(
            "Expected binary Logistic Regression."
        )

    print("\n[MODEL]")
    print("Name: Binary Logistic Regression B")
    print("Original features:", len(features))
    print("Threshold:", threshold)

    # ----------------------------------------
    # STEP 2: Load TRAINING background
    # ----------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    missing_features = [
        feature
        for feature in features
        if feature not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    background = train_df[features].sample(
        n=min(BACKGROUND_SIZE, len(train_df)),
        random_state=RANDOM_STATE,
    )

    # Use the already-fitted preprocessor.
    # Do not fit it again.

    transformed = to_dense(
        preprocessor.transform(background)
    )

    transformed = np.asarray(
        transformed,
        dtype=float,
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature count mismatch."
        )

    print("\n[BACKGROUND]")
    print("Training records:", len(background))
    print("Transformed features:", transformed.shape[1])

    # ----------------------------------------
    # STEP 3: Calculate reference statistics
    # ----------------------------------------

    reference_mean = transformed.mean(axis=0)

    coefficients = np.asarray(
        classifier.coef_[0],
        dtype=float,
    )

    intercept = float(
        classifier.intercept_[0]
    )

    if len(reference_mean) != len(coefficients):
        raise ValueError(
            "Coefficient and feature count mismatch."
        )

    base_log_odds = float(
        intercept
        + np.dot(
            coefficients,
            reference_mean,
        )
    )

    print("\n[REFERENCE]")
    print("Base log-odds:", f"{base_log_odds:.6f}")
    print(
        "Base probability:",
        f"{expit(base_log_odds):.6f}",
    )

    # ----------------------------------------
    # STEP 4: Verify against real SHAP values
    # ----------------------------------------

    print("\n[STEP] Verifying SHAP reference...")

    # IMPORTANT FIX:
    # Use ALL background records.
    # SHAP's Independent masker otherwise
    # may reduce the background to 100 rows.

    masker = shap.maskers.Independent(
        transformed,
        max_samples=len(transformed),
    )

    explainer = shap.Explainer(
        classifier,
        masker,
        algorithm="linear",
    )

    sample = transformed[:5]

    explanation = explainer(sample)

    # For a linear Logistic Regression model,
    # SHAP values in log-odds space are:
    #
    # (x - mean_background) * coefficient

    manual_values = (
        sample - reference_mean
    ) * coefficients

    shap_values = np.asarray(
        explanation.values,
        dtype=float,
    )

    shap_base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    )

    if shap_values.shape != manual_values.shape:
        raise ValueError(
            "SHAP contribution dimensions mismatch."
        )

    shap_difference = float(
        np.max(
            np.abs(
                manual_values - shap_values
            )
        )
    )

    base_difference = float(
        np.max(
            np.abs(
                shap_base_values - base_log_odds
            )
        )
    )

    # ----------------------------------------
    # STEP 5: Verify model probabilities
    # ----------------------------------------

    reconstructed_probabilities = expit(
        base_log_odds
        + manual_values.sum(axis=1)
    )

    direct_probabilities = (
        classifier.predict_proba(sample)[:, 1]
    )

    probability_difference = float(
        np.max(
            np.abs(
                reconstructed_probabilities
                - direct_probabilities
            )
        )
    )

    print(
        "SHAP difference:",
        f"{shap_difference:.10f}",
    )

    print(
        "Base value difference:",
        f"{base_difference:.10f}",
    )

    print(
        "Probability difference:",
        f"{probability_difference:.10f}",
    )

    tolerance = 1e-5

    if max(
        shap_difference,
        base_difference,
        probability_difference,
    ) > tolerance:

        raise ValueError(
            "SHAP reference verification failed."
        )

    print("[VERIFIED] SHAP values match.")
    print("[VERIFIED] Model probabilities match.")

    # ----------------------------------------
    # STEP 6: Save reference statistics
    # ----------------------------------------

    reference = {
        "model": "binary_warning_B",
        "features": features,
        "transformed_feature_names": feature_names,
        "reference_mean": reference_mean,
        "base_log_odds": base_log_odds,
        "background_size": len(background),
        "random_state": RANDOM_STATE,
        "threshold": threshold,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        reference,
        OUTPUT_PATH,
    )

    print("\n[SAVED]", OUTPUT_PATH)

    # ----------------------------------------
    # FINAL
    # ----------------------------------------

    print("\n====================================")
    print("SHAP REFERENCE COMPLETED")
    print("====================================")

    print("[SUCCESS] Reference created.")
    print("[SUCCESS] SHAP verified.")
    print("[SUCCESS] Probabilities verified.")
    print("[INFO] No raw student rows saved.")
    print("[INFO] Frozen model unchanged.")
    print("[INFO] Final Test not used.")


if __name__ == "__main__":
    main()
