
from pathlib import Path
from datetime import datetime

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

TABLES_DIR = ROOT / "reports" / "tables"
REPORTS_DIR = ROOT / "reports"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FILE HELPERS
# ==========================================

def read_csv(filename, required=True):

    path = TABLES_DIR / filename

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(f"[WARNING] Optional file missing: {filename}")
        return None

    df = pd.read_csv(path)

    print(f"[LOADED] {filename} ({len(df)} rows)")

    return df


def save_csv(df, filename):

    path = TABLES_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[SAVED] {path}")

    return path


# ==========================================
# MULTICLASS RESULTS
# ==========================================

def build_multiclass_comparison():

    print("\n========== MULTICLASS COMPARISON ==========")

    sources = [
        ("logistic_validation_results.csv", "baseline"),
        ("random_forest_validation_results.csv", "baseline"),
        ("xgboost_validation_results.csv", "baseline"),
    ]

    frames = []

    for filename, experiment in sources:

        df = read_csv(filename)

        df["experiment"] = experiment

        frames.append(df)

    # Logistic class-weight experiments
    weights = read_csv(
        "logistic_class_weight_results.csv"
    )

    for _, row in weights.iterrows():

        if row["weight_config"] == "baseline":
            continue

        frames.append(pd.DataFrame([{
            "model": f"Logistic Regression {row['stage']}",
            "stage": row["stage"],
            "experiment": (
                f"class_weight_{row['weight_config']}"
            ),
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "dropout_recall": row["dropout_recall"],
            "log_loss": row["log_loss"],
        }]))

    # Hyperparameter tuning experiments
    cv_sources = [
        ("logistic_cv_summary.csv", "Logistic Regression"),
        ("random_forest_cv_summary.csv", "Random Forest"),
        ("xgboost_cv_summary.csv", "XGBoost"),
    ]

    for filename, algorithm in cv_sources:

        df = read_csv(filename)

        for _, row in df.iterrows():

            frames.append(pd.DataFrame([{
                "model": f"{algorithm} {row['stage']}",
                "stage": row["stage"],
                "experiment": "cv_tuned",
                "accuracy": row["validation_accuracy"],
                "macro_f1": row["validation_macro_f1"],
                "dropout_recall": row["dropout_recall"],
                "log_loss": row["log_loss"],
            }]))

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    required_columns = [
        "model",
        "stage",
        "experiment",
        "accuracy",
        "macro_f1",
        "dropout_recall",
        "log_loss",
    ]

    combined = combined[required_columns].copy()

    for column in [
        "accuracy",
        "macro_f1",
        "dropout_recall",
        "log_loss",
    ]:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="raise",
        )

    if combined[required_columns].isna().any().any():
        raise ValueError(
            "Missing values in multiclass comparison."
        )

    combined = combined.sort_values(
        ["stage", "macro_f1"],
        ascending=[True, False],
    ).reset_index(drop=True)

    save_csv(
        combined,
        "unified_multiclass_comparison.csv",
    )

    print("\nBEST MACRO F1 BY STAGE:")

    winners = []

    for stage in ["A", "B"]:

        subset = combined[
            combined["stage"] == stage
        ]

        best = subset.iloc[0]

        winners.append(best.to_dict())

        print(
            f"Stage {stage}: "
            f"{best['model']} "
            f"({best['experiment']}) | "
            f"Macro F1 = {best['macro_f1']:.4f}"
        )

    return combined, pd.DataFrame(winners)


# ==========================================
# BINARY BASELINE
# ==========================================

def build_binary_comparison():

    print("\n========== BINARY BASELINE ==========")

    binary = read_csv(
        "binary_baseline_results.csv"
    )

    required = [
        "model",
        "stage",
        "accuracy",
        "dropout_precision",
        "dropout_recall",
        "dropout_f1",
        "pr_auc_ap",
        "roc_auc",
    ]

    missing = [
        col for col in required
        if col not in binary.columns
    ]

    if missing:
        raise ValueError(
            f"Missing binary metrics: {missing}"
        )

    binary = binary.sort_values(
        ["stage", "pr_auc_ap"],
        ascending=[True, False],
    ).reset_index(drop=True)

    save_csv(
        binary,
        "unified_binary_baseline.csv",
    )

    return binary


# ==========================================
# THRESHOLD COMPARISON
# ==========================================

def build_threshold_comparison():

    print("\n========== THRESHOLD COMPARISON ==========")

    logistic = read_csv(
        "binary_threshold_validation_comparison.csv"
    )

    logistic["model"] = "logistic"
    logistic["stage"] = "B"

    trees = read_csv(
        "binary_tree_thresholds_summary.csv"
    )

    combined = pd.concat(
        [logistic, trees],
        ignore_index=True,
    )

    required = [
        "model",
        "stage",
        "method",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    combined = combined[required].copy()

    combined = combined.sort_values(
        ["method", "f2"],
        ascending=[True, False],
    ).reset_index(drop=True)

    save_csv(
        combined,
        "unified_threshold_comparison.csv",
    )

    print("\nSELECTED THRESHOLD RESULTS:")

    selected = combined[
        combined["method"] == "selected_oof"
    ].copy()

    print(
        selected[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f2",
                "tp",
                "fp",
            ]
        ].round(4).to_string(index=False)
    )

    return combined, selected


# ==========================================
# CALIBRATION COMPARISON
# ==========================================

def build_calibration_comparison():

    print("\n========== CALIBRATION COMPARISON ==========")

    probabilities = read_csv(
        "binary_calibration_probability_results.csv"
    )

    thresholds = read_csv(
        "binary_calibration_threshold_results.csv"
    )

    save_csv(
        probabilities,
        "unified_calibration_probability.csv",
    )

    save_csv(
        thresholds,
        "unified_calibration_threshold.csv",
    )

    return probabilities, thresholds


# ==========================================
# MARKDOWN REPORT
# ==========================================

def markdown_table(df, columns=None):

    if columns is not None:
        df = df[columns]

    formatted = df.copy()

    for column in formatted.columns:

        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda x: f"{x:.4f}"
            )

    headers = list(formatted.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for values in formatted.itertuples(
        index=False,
        name=None,
    ):
        lines.append(
            "| "
            + " | ".join(str(v) for v in values)
            + " |"
        )

    return "\n".join(lines)


def generate_report(
    multiclass,
    multiclass_winners,
    binary,
    thresholds,
    selected_thresholds,
    calibration_probabilities,
    calibration_thresholds,
):

    print("\n========== GENERATING REPORT ==========")

    date = datetime.now().strftime("%Y-%m-%d")

    report = []

    report.append("# AcademicRisk AI - Model Comparison Report")
    report.append("")
    report.append(f"Generated: {date}")
    report.append("")
    report.append(
        "Evaluation dataset: Validation split (664 students)."
    )
    report.append(
        "Test split has not been used in this report."
    )
    report.append("")

    report.append("## 1. Multiclass Classification")
    report.append("")
    report.append(
        "Primary selection metric: Macro F1."
    )
    report.append("")

    report.append(
        markdown_table(
            multiclass,
            [
                "model",
                "stage",
                "experiment",
                "accuracy",
                "macro_f1",
                "dropout_recall",
                "log_loss",
            ],
        )
    )

    report.append("")
    report.append("### Best Macro F1 by Stage")
    report.append("")
    report.append(
        markdown_table(
            multiclass_winners,
            [
                "model",
                "stage",
                "experiment",
                "macro_f1",
                "dropout_recall",
            ],
        )
    )

    report.append("")
    report.append("## 2. Binary Classification")
    report.append("")
    report.append(
        "Risk ranking metric: Average Precision (AP)."
    )
    report.append("")

    report.append(
        markdown_table(
            binary,
            [
                "model",
                "stage",
                "accuracy",
                "dropout_precision",
                "dropout_recall",
                "dropout_f1",
                "pr_auc_ap",
                "roc_auc",
            ],
        )
    )

    report.append("")
    report.append("## 3. Threshold Optimization")
    report.append("")
    report.append(
        "Thresholds were selected using out-of-fold "
        "training predictions, targeting high F2 "
        "subject to OOF Precision >= 0.60."
    )
    report.append("")

    report.append(
        markdown_table(
            thresholds,
            [
                "model",
                "method",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f2",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        )
    )

    report.append("")
    report.append("### Selected Thresholds")
    report.append("")
    report.append(
        markdown_table(
            selected_thresholds,
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f2",
                "tp",
                "fp",
            ],
        )
    )

    report.append("")
    report.append("## 4. Probability Calibration")
    report.append("")
    report.append(
        "Lower Brier Score and Log Loss are better."
    )
    report.append("")

    report.append(
        markdown_table(calibration_probabilities)
    )

    report.append("")
    report.append("### Calibration Threshold Comparison")
    report.append("")
    report.append(
        markdown_table(
            calibration_thresholds,
            [
                "model",
                "method",
                "threshold",
                "precision",
                "recall",
                "f2",
                "tp",
                "fp",
            ],
        )
    )

    report.append("")
    report.append("## 5. Model Selection Considerations")
    report.append("")
    report.append(
        "- Multiclass models are ranked primarily "
        "by validation Macro F1."
    )
    report.append(
        "- Binary ranking quality is compared "
        "using Average Precision."
    )
    report.append(
        "- Binary alert decisions are compared "
        "using F2, precision, recall, and alert counts."
    )
    report.append(
        "- Precision >= 0.60 was an OOF threshold "
        "selection target, not a validation guarantee."
    )
    report.append(
        "- Binary and multiclass accuracy values "
        "must not be treated as results of the same task."
    )
    report.append(
        "- The Enrolled class remains challenging "
        "in the multiclass task."
    )
    report.append(
        "- Model selection is provisional until "
        "the held-out test evaluation."
    )
    report.append("")

    report.append("## 6. Evaluation Limitations")
    report.append("")
    report.append(
        "- The dataset represents historical "
        "academic outcomes and does not prove causality."
    )
    report.append(
        "- Repeated model comparisons on validation "
        "can introduce selection bias."
    )
    report.append(
        "- No prospective deployment evaluation "
        "has been performed."
    )
    report.append(
        "- Alerts are intended for supportive review, "
        "not automated punitive decisions."
    )

    report_path = (
        REPORTS_DIR /
        "AcademicRisk_AI_Model_Comparison.md"
    )

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("[SAVED REPORT]", report_path)

    return report_path


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - UNIFIED COMPARISON")
    print("====================================")

    multiclass, multiclass_winners = (
        build_multiclass_comparison()
    )

    binary = build_binary_comparison()

    thresholds, selected = (
        build_threshold_comparison()
    )

    cal_prob, cal_threshold = (
        build_calibration_comparison()
    )

    path = generate_report(
        multiclass=multiclass,
        multiclass_winners=multiclass_winners,
        binary=binary,
        thresholds=thresholds,
        selected_thresholds=selected,
        calibration_probabilities=cal_prob,
        calibration_thresholds=cal_threshold,
    )

    print("\n====================================")
    print("UNIFIED COMPARISON COMPLETED")
    print("====================================")

    print("Multiclass experiments:", len(multiclass))
    print("Binary experiments:", len(binary))
    print("Threshold results:", len(thresholds))
    print("Calibration models:", len(cal_prob))

    print("\nMain report:", path)

    print("\n[SUCCESS] All comparisons completed.")


if __name__ == "__main__":
    main()
