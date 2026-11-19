
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from scipy.special import expit


# ============================================
# PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "splits" / "validation.csv"

MODEL_PATH = (
    ROOT / "models" / "binary_logistic_B_threshold.joblib"
)

FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

RANDOM_STATE = 42
BACKGROUND_SIZE = 150
EXPLAIN_SIZE = 200


# ============================================
# READABLE FEATURE NAMES
# ============================================

FRIENDLY_NAMES = {
    "Curricular units 1st sem (approved)":
        "1st semester approved units",

    "Curricular units 1st sem (enrolled)":
        "1st semester enrolled units",

    "Curricular units 1st sem (credited)":
        "1st semester credited units",

    "Curricular units 1st sem (grade)":
        "1st semester average grade (0-20)",

    "Curricular units 1st sem (evaluations)":
        "1st semester evaluations",

    "Curricular units 1st sem (without evaluations)":
        "1st semester units without evaluations",

    "Age at enrollment":
        "Age at enrollment (years)",

    "Admission grade":
        "Admission grade (0-200)",

    "Application order":
        "Application order",

    "Course":
        "Course",

    "Application mode":
        "Application mode",

    "Previous qualification":
        "Previous qualification",
}


def readable_name(name):
    return FRIENDLY_NAMES.get(name, name)


def format_original_value(value):

    if pd.isna(value):
        return "Missing"

    if isinstance(value, (int, float, np.number)):
        return f"{float(value):g}"

    return str(value)


# ============================================
# TRANSFORMED FEATURE MAPPING
# ============================================

def make_readable_features(
    transformed_names,
    original_row,
    transformed_row,
    original_features,
):
    """
    Build readable waterfall labels without
    changing the model's transformed features.

    Numerical features: show their raw values.
    One-hot features: show whether the category
    is active, using the transformed 0/1 value.
    """

    readable_names = []
    display_values = []

    # Longest column names first to avoid
    # ambiguous prefix matching.
    columns = sorted(
        original_features,
        key=len,
        reverse=True,
    )

    for i, transformed_name in enumerate(
        transformed_names
    ):

        name = str(transformed_name)

        if name.startswith("numeric__"):

            original_name = name[len("numeric__"):]

            readable_names.append(
                readable_name(original_name)
            )

            if original_name in original_row.index:
                display_values.append(
                    format_original_value(
                        original_row[original_name]
                    )
                )
            else:
                display_values.append("Unknown")

        elif name.startswith("categorical__"):

            encoded_name = name[len("categorical__"):]

            source_column = None
            category = None

            for column in columns:

                prefix = f"{column}_"

                if encoded_name.startswith(prefix):

                    source_column = column
                    category = encoded_name[len(prefix):]
                    break

            if source_column is None:

                readable_names.append(encoded_name)

            else:

                readable_names.append(
                    f"{readable_name(source_column)}: "
                    f"{category}"
                )

            # These values are the actual
            # one-hot encoded feature states.
            encoded_value = float(
                transformed_row[i]
            )

            if np.isclose(encoded_value, 1.0):
                display_values.append("Selected")

            elif np.isclose(encoded_value, 0.0):
                display_values.append("Not selected")

            else:
                display_values.append(
                    f"{encoded_value:.3f}"
                )

        else:

            readable_names.append(name)

            display_values.append(
                format_original_value(
                    transformed_row[i]
                )
            )

    return readable_names, np.asarray(
        display_values,
        dtype=object,
    )


# ============================================
# MAIN
# ============================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - READABLE SHAP")
    print("====================================")

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # 1. Load frozen model
    # ----------------------------------------

    artifact = joblib.load(MODEL_PATH)

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
            "Expected binary classes [0, 1]."
        )

    print("\n[MODEL]")
    print("Features:", len(features))
    print("Threshold:", threshold)

    # ----------------------------------------
    # 2. Load data
    # ----------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    background_df = train_df[features].sample(
        n=min(BACKGROUND_SIZE, len(train_df)),
        random_state=RANDOM_STATE,
    )

    explain_df = validation_df[features].sample(
        n=min(EXPLAIN_SIZE, len(validation_df)),
        random_state=RANDOM_STATE,
    )

    X_background = preprocessor.transform(
        background_df
    )

    X_explain = preprocessor.transform(
        explain_df
    )

    if hasattr(X_background, "toarray"):
        X_background = X_background.toarray()

    if hasattr(X_explain, "toarray"):
        X_explain = X_explain.toarray()

    X_background = np.asarray(
        X_background,
        dtype=float,
    )

    X_explain = np.asarray(
        X_explain,
        dtype=float,
    )

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    if len(transformed_names) != X_explain.shape[1]:
        raise ValueError(
            "Transformed feature name mismatch."
        )

    print("\n[DATA]")
    print("Training background:", len(background_df))
    print("Validation records:", len(explain_df))
    print("Transformed features:", X_explain.shape[1])

    # ----------------------------------------
    # 3. Calculate SHAP
    # ----------------------------------------

    print("\n[STEP 1] Calculating SHAP...")

    explainer = shap.Explainer(
        classifier,
        X_background,
        algorithm="linear",
    )

    explanation = explainer(X_explain)

    values = np.asarray(
        explanation.values
    )

    base_values = np.asarray(
        explanation.base_values
    )

    probabilities = pipeline.predict_proba(
        explain_df
    )[:, 1]

    reconstructed = expit(
        base_values + values.sum(axis=1)
    )

    difference = float(
        np.max(
            np.abs(
                probabilities - reconstructed
            )
        )
    )

    print(
        "Maximum reconstruction error:",
        f"{difference:.10f}",
    )

    if difference > 1e-5:
        raise ValueError(
            "SHAP verification failed."
        )

    print("[VERIFIED] SHAP matches predictions.")

    # ----------------------------------------
    # 4. Select example students
    # ----------------------------------------

    selected_cases = {
        "highest_risk": int(
            np.argmax(probabilities)
        ),

        "lowest_risk": int(
            np.argmin(probabilities)
        ),

        "near_threshold": int(
            np.argmin(
                np.abs(
                    probabilities - threshold
                )
            )
        ),
    }

    summary_rows = []
    contribution_rows = []

    # ----------------------------------------
    # 5. Generate readable waterfalls
    # ----------------------------------------

    print("\n[STEP 2] Generating readable plots...")

    for case_name, idx in selected_cases.items():

        original_row = explain_df.iloc[idx]

        transformed_row = X_explain[idx]

        readable_names, display_values = (
            make_readable_features(
                transformed_names=transformed_names,
                original_row=original_row,
                transformed_row=transformed_row,
                original_features=features,
            )
        )

        # Create a separate display explanation.
        # The SHAP values and model inputs
        # remain exactly the same.

        readable_explanation = shap.Explanation(
            values=values[idx],
            base_values=float(
                base_values[idx]
            ),
            data=transformed_row,
            display_data=display_values,
            feature_names=readable_names,
        )

        probability = float(
            probabilities[idx]
        )

        warning = int(
            probability >= threshold
        )

        print(f"\nCASE: {case_name}")
        print(f"Dropout probability: {probability:.4f}")
        print(f"Warning: {warning}")

        # Print important original values

        important_fields = [
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (grade)",
            "Age at enrollment",
            "Admission grade",
            "Course",
            "Application mode",
        ]

        print("\nORIGINAL STUDENT VALUES:")

        for field in important_fields:

            if field in original_row.index:

                value = format_original_value(
                    original_row[field]
                )

                print(f"  {field}: {value}")

        # Save readable waterfall

        plt.close("all")

        shap.plots.waterfall(
            readable_explanation,
            max_display=12,
            show=False,
        )

        plt.title(
            f"{case_name.replace('_', ' ').title()}\n"
            f"Dropout probability: {probability:.2%} | "
            f"Threshold: {threshold:.0%}",
            fontsize=11,
            pad=15,
        )

        output_path = (
            FIGURE_DIR /
            f"binary_B_shap_readable_{case_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close("all")

        print(f"[SAVED FIGURE] {output_path}")

        # Save original student record summary

        summary = {
            "case": case_name,
            "validation_row_index": int(
                explain_df.index[idx]
            ),
            "dropout_probability": probability,
            "threshold": threshold,
            "warning": warning,
        }

        for field in important_fields:

            if field in original_row.index:

                summary[field] = (
                    original_row[field]
                )

        summary_rows.append(summary)

        # Save ALL individual feature contributions

        for i, feature_name in enumerate(
            readable_names
        ):

            contribution_rows.append({
                "case": case_name,
                "feature": feature_name,
                "display_value": display_values[i],
                "transformed_value": float(
                    transformed_row[i]
                ),
                "shap_log_odds": float(
                    values[idx, i]
                ),
            })

    # ----------------------------------------
    # 6. Export tables
    # ----------------------------------------

    summary_path = (
        TABLE_DIR /
        "binary_B_shap_readable_students.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    contributions_path = (
        TABLE_DIR /
        "binary_B_shap_readable_contributions.csv"
    )

    pd.DataFrame(
        contribution_rows
    ).to_csv(
        contributions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[SAVED TABLE] {summary_path}")
    print(f"[SAVED TABLE] {contributions_path}")

    # ----------------------------------------
    # FINISH
    # ----------------------------------------

    print("\n====================================")
    print("READABLE SHAP COMPLETED")
    print("====================================")

    print("[SUCCESS] Original values displayed.")
    print("[SUCCESS] SHAP values unchanged.")
    print("[SUCCESS] Frozen model unchanged.")
    print("[INFO] Only Training/Validation used.")


if __name__ == "__main__":
    main()
