
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from scipy.special import expit


# ============================================
# PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "splits" / "validation.csv"

MODEL_PATH = (
    ROOT / "models" / "binary_logistic_B_threshold.joblib"
)

TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
BACKGROUND_SIZE = 150
EXPLAIN_SIZE = 200


# ============================================
# HELPERS
# ============================================

def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def get_feature_names(preprocessor, count):
    try:
        names = list(preprocessor.get_feature_names_out())

        if len(names) == count:
            return names

    except (AttributeError, ValueError, TypeError):
        pass

    warnings.warn(
        "Could not extract transformed feature names. "
        "Using generic names."
    )

    return [f"feature_{i}" for i in range(count)]


def save_plot(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    print(f"[SAVED FIGURE] {path}")


# ============================================
# MAIN
# ============================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - BINARY SHAP")
    print("====================================")

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # 1. Load saved model
    # ----------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    artifact = joblib.load(MODEL_PATH)

    pipeline = artifact["pipeline"]
    features = list(artifact["features"])
    threshold = float(artifact["threshold"])

    print("\n[MODEL]")
    print("File:", MODEL_PATH.name)
    print("Original features:", len(features))
    print("Threshold:", threshold)

    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise TypeError(
            "Expected a fitted linear Logistic Regression model."
        )

    if list(classifier.classes_) != [0, 1]:
        raise ValueError(
            f"Unexpected class order: {classifier.classes_}"
        )

    # ----------------------------------------
    # 2. Load TRAIN and VALIDATION
    # ----------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    missing_train = set(features) - set(train_df.columns)
    missing_val = set(features) - set(validation_df.columns)

    if missing_train or missing_val:
        raise ValueError(
            f"Missing train: {missing_train}; "
            f"missing validation: {missing_val}"
        )

    background_df = train_df[features].sample(
        n=min(BACKGROUND_SIZE, len(train_df)),
        random_state=RANDOM_STATE,
    )

    explain_df = validation_df[features].sample(
        n=min(EXPLAIN_SIZE, len(validation_df)),
        random_state=RANDOM_STATE,
    )

    # The saved preprocessor is used as-is.
    # No fit / fit_transform is performed here.

    X_background = to_dense(
        preprocessor.transform(background_df)
    )

    X_explain = to_dense(
        preprocessor.transform(explain_df)
    )

    feature_names = get_feature_names(
        preprocessor,
        X_explain.shape[1],
    )

    X_background = np.asarray(
        X_background,
        dtype=float,
    )

    X_explain = np.asarray(
        X_explain,
        dtype=float,
    )

    print("\n[DATA]")
    print("Training background:", len(background_df))
    print("Validation explained:", len(explain_df))
    print("Transformed features:", X_explain.shape[1])

    # ----------------------------------------
    # 3. SHAP Linear Explainer
    # ----------------------------------------

    print("\n[STEP 1] Computing SHAP values...")

    explainer = shap.Explainer(
        classifier,
        X_background,
        algorithm="linear",
    )

    explanation = explainer(X_explain)

    values = np.asarray(explanation.values)

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values; got {values.shape}"
        )

    if values.shape != X_explain.shape:
        raise ValueError(
            "SHAP dimensions do not match transformed data."
        )

    explanation.feature_names = feature_names

    # Logistic Regression linear SHAP explains log-odds.
    # Check that the SHAP decomposition reproduces
    # predicted probabilities.

    base_values = np.asarray(
        explanation.base_values
    )

    reconstructed_probability = expit(
        base_values + values.sum(axis=1)
    )

    actual_probability = pipeline.predict_proba(
        explain_df
    )[:, 1]

    maximum_difference = np.max(
        np.abs(
            reconstructed_probability -
            actual_probability
        )
    )

    print(
        "Maximum probability reconstruction difference:",
        f"{maximum_difference:.8f}",
    )

    if maximum_difference > 1e-5:
        raise ValueError(
            "SHAP reconstruction check failed."
        )

    print("[VERIFIED] SHAP explanations match the model.")

    # ----------------------------------------
    # 4. Global importance
    # ----------------------------------------

    print("\n[STEP 2] Global feature importance...")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_log_odds": np.abs(
            values
        ).mean(axis=0),
        "mean_shap_log_odds": values.mean(axis=0),
    })

    importance_df = importance_df.sort_values(
        "mean_abs_shap_log_odds",
        ascending=False,
    )

    importance_path = (
        TABLE_DIR / "binary_B_shap_global_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTOP 15 FEATURES:")
    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    print(f"\n[SAVED TABLE] {importance_path}")

    # ----------------------------------------
    # 5. Global bar chart
    # ----------------------------------------

    print("\n[STEP 3] Creating global plots...")

    shap.plots.bar(
        explanation,
        max_display=15,
        show=False,
    )

    save_plot(
        FIGURE_DIR / "binary_B_shap_global_bar.png"
    )

    # ----------------------------------------
    # 6. Beeswarm plot
    # ----------------------------------------

    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False,
    )

    save_plot(
        FIGURE_DIR / "binary_B_shap_beeswarm.png"
    )

    # ----------------------------------------
    # 7. Local explanations
    # ----------------------------------------

    print("\n[STEP 4] Local explanations...")

    highest_risk = int(
        np.argmax(actual_probability)
    )

    lowest_risk = int(
        np.argmin(actual_probability)
    )

    near_threshold = int(
        np.argmin(
            np.abs(actual_probability - threshold)
        )
    )

    selected_cases = {
        "highest_risk": highest_risk,
        "lowest_risk": lowest_risk,
        "near_threshold": near_threshold,
    }

    local_rows = []

    for case_name, idx in selected_cases.items():

        probability = float(
            actual_probability[idx]
        )

        warning = int(
            probability >= threshold
        )

        print(f"\nCase: {case_name}")
        print(f"Probability: {probability:.4f}")
        print(f"Warning: {warning}")

        local_rows.append({
            "case": case_name,
            "validation_row_index": int(
                explain_df.index[idx]
            ),
            "dropout_probability": probability,
            "threshold": threshold,
            "warning": warning,
        })

        plt.figure()

        shap.plots.waterfall(
            explanation[idx],
            max_display=12,
            show=False,
        )

        save_plot(
            FIGURE_DIR /
            f"binary_B_shap_{case_name}.png"
        )

    local_path = (
        TABLE_DIR / "binary_B_shap_local_examples.csv"
    )

    pd.DataFrame(local_rows).to_csv(
        local_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[SAVED TABLE] {local_path}")

    # ----------------------------------------
    # FINAL
    # ----------------------------------------

    print("\n====================================")
    print("SHAP EXPLANATION COMPLETED")
    print("====================================")

    print("[SUCCESS] Binary model explained.")
    print("[INFO] Training/Validation only.")
    print("[INFO] Final Test was not used.")
    print("[INFO] SHAP values are in log-odds units.")


if __name__ == "__main__":
    main()
