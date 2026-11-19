
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

RANDOM_STATE = 42

STAGE = "B"

DEFAULT_THRESHOLD = 0.50
MIN_PRECISION = 0.60

THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.01),
    2,
)

VALID_TARGETS = {
    "Dropout",
    "Enrolled",
    "Graduate",
}


# ==========================================
# TARGET CONVERSION
# ==========================================

def make_binary_target(series):

    if series.isna().any():
        raise ValueError("Target contains missing values.")

    unexpected = (
        set(series.unique()) - VALID_TARGETS
    )

    if unexpected:
        raise ValueError(
            f"Unexpected target classes: {unexpected}"
        )

    return (series == "Dropout").astype(int)


# ==========================================
# MODEL
# ==========================================

def build_model(config, features):

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="linear",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


# ==========================================
# THRESHOLD METRICS
# ==========================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ==========================================
# SELECT THRESHOLD ON TRAINING ONLY
# ==========================================

def select_threshold(
    y_train,
    oof_probabilities,
):

    results = []

    for threshold in THRESHOLDS:

        metrics = evaluate_threshold(
            y_true=y_train,
            probabilities=oof_probabilities,
            threshold=threshold,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLES_DIR /
        "binary_threshold_oof_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = results_df[
        (results_df["precision"] >= MIN_PRECISION)
        & (results_df["tp"] > 0)
    ].copy()

    print("\n========== THRESHOLD SELECTION ==========")

    print(
        f"Minimum OOF Precision: {MIN_PRECISION:.2f}"
    )

    if eligible.empty:

        print(
            "[WARNING] No threshold satisfied "
            "the minimum precision requirement."
        )

        print(
            "[INFO] Keeping the default threshold."
        )

        selected = DEFAULT_THRESHOLD
        selection_status = "fallback_default"

    else:

        eligible = eligible.sort_values(
            by=["f2", "precision", "threshold"],
            ascending=[False, False, False],
        )

        best = eligible.iloc[0]

        selected = float(best["threshold"])
        selection_status = "optimized"

        print(
            f"Selected threshold: {selected:.2f}"
        )

        print(
            f"OOF Precision: {best['precision']:.4f}"
        )

        print(
            f"OOF Recall:    {best['recall']:.4f}"
        )

        print(
            f"OOF F2:        {best['f2']:.4f}"
        )

    print("[SAVED OOF RESULTS]", output_path)

    return selected, selection_status


# ==========================================
# PRINT METRICS
# ==========================================

def print_metrics(title, metrics):

    print(f"\n========== {title} ==========")

    for key in [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
    ]:
        print(f"{key:<12}: {metrics[key]:.4f}")

    print("\nConfusion matrix values:")
    print("TP:", metrics["tp"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])
    print("TN:", metrics["tn"])


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - THRESHOLD OPTIMIZATION")
    print("====================================")

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    features = get_features(config, STAGE)

    target = config["target"]

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = make_binary_target(
        train_df[target]
    )

    y_val = make_binary_target(
        val_df[target]
    )

    print("\nModel: Binary Logistic Regression")
    print("Stage:", STAGE)
    print("Features:", len(features))
    print("Training students:", len(X_train))
    print("Validation students:", len(X_val))

    # ======================================
    # STEP 1: OUT-OF-FOLD PROBABILITIES
    # ======================================

    model = build_model(
        config=config,
        features=features,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\n[STEP 1] Generating OOF probabilities...")

    # Each student's probability comes from
    # a model trained without that student's row.
    # Preprocessing is fitted inside each fold.

    oof_probabilities_all = cross_val_predict(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=2,
    )

    # Binary classes are ordered [0, 1].
    oof_probabilities = oof_probabilities_all[:, 1]

    # ======================================
    # STEP 2: SELECT DECISION THRESHOLD
    # ======================================

    print("\n[STEP 2] Searching thresholds...")

    selected_threshold, selection_status = (
        select_threshold(
            y_train=y_train,
            oof_probabilities=oof_probabilities,
        )
    )

    # ======================================
    # STEP 3: TRAIN FINAL CANDIDATE
    # ======================================

    print("\n[STEP 3] Fitting on all training rows...")

    final_pipeline = clone(model)

    final_pipeline.fit(
        X_train,
        y_train,
    )

    # ======================================
    # STEP 4: VALIDATION EVALUATION
    # ======================================

    print("\n[STEP 4] Evaluating on Validation...")

    val_probabilities = (
        final_pipeline.predict_proba(X_val)[:, 1]
    )

    baseline = evaluate_threshold(
        y_true=y_val,
        probabilities=val_probabilities,
        threshold=DEFAULT_THRESHOLD,
    )

    optimized = evaluate_threshold(
        y_true=y_val,
        probabilities=val_probabilities,
        threshold=selected_threshold,
    )

    print_metrics(
        "DEFAULT THRESHOLD 0.50",
        baseline,
    )

    print_metrics(
        "SELECTED THRESHOLD",
        optimized,
    )

    print("\n========== PROBABILITY METRICS ==========")

    print(
        "Average Precision:",
        round(
            average_precision_score(
                y_val,
                val_probabilities,
            ),
            4,
        ),
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_val,
                val_probabilities,
            ),
            4,
        ),
    )

    predictions = (
        val_probabilities >= selected_threshold
    ).astype(int)

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=[
                "Not Dropout",
                "Dropout",
            ],
            zero_division=0,
        )
    )

    # ======================================
    # STEP 5: SAVE RESULTS
    # ======================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR /
        "binary_logistic_B_threshold.joblib"
    )

    joblib.dump(
        {
            "pipeline": final_pipeline,
            "features": features,
            "stage": STAGE,
            "positive_label": 1,
            "threshold": selected_threshold,
            "threshold_selection": selection_status,
            "min_oof_precision": MIN_PRECISION,
        },
        model_path,
    )

    comparison = pd.DataFrame([
        {
            "method": "default_0.50",
            **baseline,
        },
        {
            "method": "selected_oof",
            **optimized,
        },
    ])

    comparison_path = (
        TABLES_DIR /
        "binary_threshold_validation_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== COMPARISON ==========")

    print(
        comparison[
            [
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
        ].round(4).to_string(index=False)
    )

    print("\n[SAVED MODEL]", model_path)
    print("[SAVED COMPARISON]", comparison_path)

    print(
        "\n[SUCCESS] Threshold optimization completed."
    )


if __name__ == "__main__":
    main()
