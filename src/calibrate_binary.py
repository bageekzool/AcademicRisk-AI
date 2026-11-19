
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
)
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
    brier_score_loss,
    log_loss,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)

from src.train_binary import make_binary_target


# ==========================================
# PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42

STAGE = "B"

MIN_PRECISION = 0.60
DEFAULT_THRESHOLD = 0.50

THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.01),
    2,
)


# ==========================================
# BUILD ORIGINAL MODEL
# ==========================================

def build_base_model(config, features):

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="linear",
    )

    return Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ==========================================
# BUILD CALIBRATED MODEL
# ==========================================

def build_calibrated_model(base_model):

    return CalibratedClassifierCV(
        estimator=clone(base_model),
        method="sigmoid",
        cv=3,
        n_jobs=1,
    )


# ==========================================
# EVALUATION FUNCTIONS
# ==========================================

def probability_metrics(y_true, probabilities):

    return {
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
    }


def threshold_metrics(y_true, probabilities, threshold):

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
# THRESHOLD SELECTION FROM TRAINING OOF
# ==========================================

def select_threshold(name, y_true, probabilities):

    rows = [
        threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ]

    results = pd.DataFrame(rows)

    path = (
        TABLES_DIR /
        f"calibration_{name}_oof_thresholds.csv"
    )

    results.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = results[
        (results["precision"] >= MIN_PRECISION)
        & (results["tp"] > 0)
    ].copy()

    if eligible.empty:

        print(
            f"[WARNING] {name}: no threshold "
            "satisfies minimum OOF precision."
        )

        return DEFAULT_THRESHOLD, "fallback_default"

    best = (
        eligible.sort_values(
            ["f2", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    )

    threshold = float(best["threshold"])

    print(f"\n{name.upper()} OOF THRESHOLD:")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f}")
    print(f"F2:        {best['f2']:.4f}")

    return threshold, "optimized"


# ==========================================
# CALIBRATION PLOT
# ==========================================

def save_calibration_plot(
    y_val,
    raw_probabilities,
    calibrated_probabilities,
):

    fig, ax = plt.subplots(figsize=(8, 7))

    CalibrationDisplay.from_predictions(
        y_val,
        raw_probabilities,
        n_bins=10,
        strategy="quantile",
        name="Original Logistic Regression",
        ax=ax,
    )

    CalibrationDisplay.from_predictions(
        y_val,
        calibrated_probabilities,
        n_bins=10,
        strategy="quantile",
        name="Sigmoid Calibrated",
        ax=ax,
    )

    ax.set_title(
        "AcademicRisk AI - Probability Calibration"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    figure_path = (
        FIGURES_DIR /
        "binary_logistic_B_calibration.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("[SAVED FIGURE]", figure_path)


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - PROBABILITY CALIBRATION")
    print("====================================")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    features = get_features(config, STAGE)

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = make_binary_target(
        train_df[config["target"]]
    )

    y_val = make_binary_target(
        val_df[config["target"]]
    )

    print("\nStage:", STAGE)
    print("Features:", len(features))
    print("Training students:", len(X_train))
    print("Validation students:", len(X_val))

    base_model = build_base_model(
        config,
        features,
    )

    calibrated_model = build_calibrated_model(
        base_model
    )

    # ======================================
    # STEP 1: OUTER CROSS-VALIDATION
    # ======================================

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\n[STEP 1] Raw model OOF predictions...")

    raw_oof = cross_val_predict(
        estimator=clone(base_model),
        X=X_train,
        y=y_train,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=2,
    )[:, 1]

    print(
        "[STEP 2] Calibrated model OOF predictions..."
    )

    # Each outer fold trains its own calibrated model.
    # Calibration uses 3 inner folds.
    # No outer validation row is used for fitting.

    calibrated_oof = cross_val_predict(
        estimator=clone(calibrated_model),
        X=X_train,
        y=y_train,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=2,
    )[:, 1]

    # ======================================
    # STEP 3: OOF PROBABILITY METRICS
    # ======================================

    print("\n========== OOF PROBABILITY METRICS ==========")

    for name, probabilities in [
        ("raw", raw_oof),
        ("calibrated", calibrated_oof),
    ]:

        scores = probability_metrics(
            y_train,
            probabilities,
        )

        print(f"\n{name.upper()}:")

        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")

    # ======================================
    # STEP 4: SELECT THRESHOLDS
    # ======================================

    print("\n[STEP 3] Selecting OOF thresholds...")

    raw_threshold, raw_status = select_threshold(
        "raw",
        y_train,
        raw_oof,
    )

    cal_threshold, cal_status = select_threshold(
        "calibrated",
        y_train,
        calibrated_oof,
    )

    # ======================================
    # STEP 5: FIT FULL TRAINING
    # ======================================

    print("\n[STEP 4] Training final candidates...")

    final_raw = clone(base_model)
    final_cal = clone(calibrated_model)

    final_raw.fit(X_train, y_train)
    final_cal.fit(X_train, y_train)

    raw_val = final_raw.predict_proba(X_val)[:, 1]
    cal_val = final_cal.predict_proba(X_val)[:, 1]

    # ======================================
    # STEP 6: VALIDATION COMPARISON
    # ======================================

    print("\n========== VALIDATION PROBABILITY METRICS ==========")

    probability_rows = []

    for name, probabilities in [
        ("raw", raw_val),
        ("calibrated", cal_val),
    ]:

        scores = probability_metrics(
            y_val,
            probabilities,
        )

        probability_rows.append({
            "model": name,
            **scores,
        })

    probability_df = pd.DataFrame(
        probability_rows
    )

    print(
        probability_df
        .round(4)
        .to_string(index=False)
    )

    threshold_rows = []

    candidates = [
        ("raw", raw_val, raw_threshold),
        ("calibrated", cal_val, cal_threshold),
    ]

    for name, probabilities, selected in candidates:

        for method, threshold in [
            ("default_0.50", DEFAULT_THRESHOLD),
            ("selected_oof", selected),
        ]:

            metrics = threshold_metrics(
                y_val,
                probabilities,
                threshold,
            )

            threshold_rows.append({
                "model": name,
                "method": method,
                **metrics,
            })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print("\n========== VALIDATION THRESHOLD METRICS ==========")

    print(
        threshold_df
        .round(4)
        .to_string(index=False)
    )

    # ======================================
    # STEP 7: SAVE FIGURE AND RESULTS
    # ======================================

    save_calibration_plot(
        y_val,
        raw_val,
        cal_val,
    )

    probability_path = (
        TABLES_DIR /
        "binary_calibration_probability_results.csv"
    )

    threshold_path = (
        TABLES_DIR /
        "binary_calibration_threshold_results.csv"
    )

    probability_df.to_csv(
        probability_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    raw_model_path = (
        MODELS_DIR /
        "binary_logistic_B_raw_calibration_study.joblib"
    )

    calibrated_model_path = (
        MODELS_DIR /
        "binary_logistic_B_calibrated.joblib"
    )

    joblib.dump(
        {
            "pipeline": final_raw,
            "features": features,
            "stage": STAGE,
            "threshold": raw_threshold,
            "threshold_selection": raw_status,
            "positive_label": 1,
            "calibration": "none",
        },
        raw_model_path,
    )

    joblib.dump(
        {
            "pipeline": final_cal,
            "features": features,
            "stage": STAGE,
            "threshold": cal_threshold,
            "threshold_selection": cal_status,
            "positive_label": 1,
            "calibration": "sigmoid",
        },
        calibrated_model_path,
    )

    print("\n[SAVED]", probability_path)
    print("[SAVED]", threshold_path)
    print("[SAVED MODEL]", raw_model_path)
    print("[SAVED MODEL]", calibrated_model_path)

    print(
        "\n[SUCCESS] Calibration study completed."
    )


if __name__ == "__main__":
    main()
