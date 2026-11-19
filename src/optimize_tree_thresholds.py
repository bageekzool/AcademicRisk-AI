
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
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
)

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)

from src.train_binary import (
    make_binary_target,
    build_classifier,
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

MODEL_NAMES = [
    "random_forest",
    "xgboost",
]

DEFAULT_THRESHOLD = 0.50
MIN_PRECISION = 0.60

THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.01),
    2,
)


# ==========================================
# BUILD MODEL
# ==========================================

def build_model(model_name, config, features):

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="tree",
    )

    classifier = build_classifier(model_name)

    # Avoid nested parallelism during cross-validation.
    classifier.set_params(n_jobs=1)

    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])


# ==========================================
# CALCULATE THRESHOLD METRICS
# ==========================================

def evaluate_threshold(y_true, probabilities, threshold):

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
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "f2": fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ==========================================
# SELECT THRESHOLD USING OOF PREDICTIONS
# ==========================================

def select_threshold(model_name, y_train, probabilities):

    records = [
        evaluate_threshold(
            y_train,
            probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ]

    results_df = pd.DataFrame(records)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    path = (
        TABLES_DIR
        / f"binary_{model_name}_threshold_oof.csv"
    )

    results_df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = results_df[
        (results_df["precision"] >= MIN_PRECISION)
        & (results_df["tp"] > 0)
    ]

    if eligible.empty:

        print(
            "[WARNING] No threshold met the "
            "minimum OOF precision requirement."
        )

        print("[INFO] Using default threshold 0.50.")

        return DEFAULT_THRESHOLD, "fallback_default"

    best = (
        eligible.sort_values(
            by=["f2", "precision", "threshold"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    selected = float(best["threshold"])

    print(f"Selected Threshold: {selected:.2f}")
    print(f"OOF Precision:      {best['precision']:.4f}")
    print(f"OOF Recall:         {best['recall']:.4f}")
    print(f"OOF F1:             {best['f1']:.4f}")
    print(f"OOF F2:             {best['f2']:.4f}")

    print("[SAVED OOF RESULTS]", path)

    return selected, "optimized"


# ==========================================
# PRINT VALIDATION METRICS
# ==========================================

def print_metrics(name, result):

    print(f"\n========== {name} ==========")

    for metric in [
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
    ]:

        value = result[metric]

        if metric in ["tp", "fp", "fn", "tn"]:
            print(f"{metric:<12}: {value}")
        else:
            print(f"{metric:<12}: {value:.4f}")


# ==========================================
# OPTIMIZE ONE CLASSIFIER
# ==========================================

def optimize_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    features,
):

    print("\n====================================")
    print(f"MODEL: {model_name.upper()} - {STAGE}")
    print("====================================")

    model = build_model(
        model_name,
        config,
        features,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # STEP 1: OOF probabilities
    print("\n[STEP 1] Generating OOF probabilities...")

    oof_probabilities = cross_val_predict(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=2,
    )[:, 1]

    # STEP 2: Select threshold on Training only
    print("\n[STEP 2] Selecting threshold...")

    selected_threshold, selection_status = (
        select_threshold(
            model_name,
            y_train,
            oof_probabilities,
        )
    )

    # STEP 3: Fit on all Training
    print("\n[STEP 3] Training on full Training split...")

    final_pipeline = clone(model)

    final_pipeline.fit(X_train, y_train)

    # STEP 4: Evaluate on Validation
    print("\n[STEP 4] Evaluating on Validation...")

    val_probabilities = (
        final_pipeline.predict_proba(X_val)[:, 1]
    )

    default_result = evaluate_threshold(
        y_val,
        val_probabilities,
        DEFAULT_THRESHOLD,
    )

    selected_result = evaluate_threshold(
        y_val,
        val_probabilities,
        selected_threshold,
    )

    print_metrics(
        "DEFAULT THRESHOLD 0.50",
        default_result,
    )

    print_metrics(
        "SELECTED THRESHOLD",
        selected_result,
    )

    ap = average_precision_score(
        y_val,
        val_probabilities,
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probabilities,
    )

    print("\n========== PROBABILITY METRICS ==========")
    print(f"Average Precision: {ap:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")

    # STEP 5: Save results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        MODELS_DIR
        / f"binary_{model_name}_B_threshold.joblib"
    )

    joblib.dump(
        {
            "pipeline": final_pipeline,
            "features": features,
            "stage": STAGE,
            "model_name": model_name,
            "positive_label": 1,
            "threshold": selected_threshold,
            "threshold_selection": selection_status,
            "min_oof_precision": MIN_PRECISION,
        },
        model_path,
    )

    comparison = pd.DataFrame([
        {
            "model": model_name,
            "stage": STAGE,
            "method": "default_0.50",
            **default_result,
            "average_precision": ap,
            "roc_auc": roc_auc,
        },
        {
            "model": model_name,
            "stage": STAGE,
            "method": "selected_oof",
            **selected_result,
            "average_precision": ap,
            "roc_auc": roc_auc,
        },
    ])

    comparison_path = (
        TABLES_DIR
        / f"binary_{model_name}_threshold_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[SAVED MODEL]", model_path)
    print("[SAVED COMPARISON]", comparison_path)

    return comparison


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - TREE THRESHOLDS")
    print("====================================")

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

    print("Stage:", STAGE)
    print("Features:", len(features))
    print("Training students:", len(X_train))
    print("Validation students:", len(X_val))
    print("Minimum OOF Precision:", MIN_PRECISION)

    all_results = []

    for model_name in MODEL_NAMES:

        comparison = optimize_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            features=features,
        )

        all_results.append(comparison)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        TABLES_DIR
        / "binary_tree_thresholds_summary.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n====================================")
    print("FINAL THRESHOLD COMPARISON")
    print("====================================")

    columns = [
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
    ]

    print(
        combined[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n[SAVED SUMMARY]", output_path)

    print(
        "\n[SUCCESS] Tree threshold optimization completed."
    )


if __name__ == "__main__":
    main()
