
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

RANDOM_STATE = 42

VALID_TARGETS = {
    "Dropout",
    "Enrolled",
    "Graduate",
}

LABEL_NAMES = ["Not Dropout", "Dropout"]


# ==========================================
# BINARY TARGET
# ==========================================

def make_binary_target(series):

    unexpected = set(series.dropna().unique()) - VALID_TARGETS

    if unexpected or series.isna().any():
        raise ValueError(
            f"Unexpected or missing target values: {unexpected}"
        )

    # Dropout = 1
    # Enrolled / Graduate = 0

    return (series == "Dropout").astype(int)


# ==========================================
# BUILD CLASSIFIER
# ==========================================

def build_classifier(model_name):

    if model_name == "logistic":

        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "xgboost":

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ==========================================
# TRAIN ONE MODEL
# ==========================================

def train_binary_model(
    model_name,
    stage,
    train_df,
    val_df,
    config,
):

    print(
        f"\n========== "
        f"{model_name.upper()} - MODEL {stage} "
        f"=========="
    )

    features = get_features(config, stage)

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = make_binary_target(
        train_df[config["target"]]
    )

    y_val = make_binary_target(
        val_df[config["target"]]
    )

    model_type = (
        "linear"
        if model_name == "logistic"
        else "tree"
    )

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type=model_type,
    )

    classifier = build_classifier(model_name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    print("[INFO] Training binary model...")

    # Training only
    pipeline.fit(X_train, y_train)

    # Validation only
    predictions = pipeline.predict(X_val)

    # Positive-class probability (Dropout = 1)
    classes = pipeline.named_steps["classifier"].classes_

    positive_index = list(classes).index(1)

    probabilities = pipeline.predict_proba(X_val)[
        :, positive_index
    ]

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_val, predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val, predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities,
    )

    loss = log_loss(
        y_val,
        probabilities,
        labels=[0, 1],
    )

    print("\n========== METRICS ==========")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Dropout Precision: {precision:.4f}")
    print(f"Dropout Recall:    {recall:.4f}")
    print(f"Dropout F1:        {f1:.4f}")
    print(f"PR-AUC (AP):       {pr_auc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Log Loss:          {loss:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("\n========== CONFUSION MATRIX ==========")

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    )

    print("Rows = Actual, Columns = Predicted")
    print(matrix_df.to_string())

    tn, fp, fn, tp = matrix.ravel()

    print("\n========== DROPOUT DETECTION ==========")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")

    # ======================================
    # SAVE MODEL
    # ======================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR /
        f"binary_{model_name}_{stage}.joblib"
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": features,
            "stage": stage,
            "model_name": model_name,
            "negative_class": "Not Dropout",
            "positive_class": "Dropout",
            "positive_label": 1,
            "threshold": 0.5,
        },
        model_path,
    )

    print("\n[SAVED MODEL]", model_path)

    return {
        "model": model_name,
        "stage": stage,
        "features": len(features),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "dropout_precision": precision,
        "dropout_recall": recall,
        "dropout_f1": f1,
        "pr_auc_ap": pr_auc,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - BINARY CLASSIFICATION")
    print("====================================")

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    print("\nTraining students:", len(train_df))
    print("Validation students:", len(val_df))

    train_target = make_binary_target(
        train_df[config["target"]]
    )

    val_target = make_binary_target(
        val_df[config["target"]]
    )

    print("\n========== BINARY CLASS DISTRIBUTION ==========")

    print("TRAIN:")
    print(train_target.value_counts().sort_index())

    print("\nVALIDATION:")
    print(val_target.value_counts().sort_index())

    print("\nLabel mapping:")
    print("0 = Not Dropout")
    print("1 = Dropout")

    prevalence = val_target.mean()

    print(
        f"\nValidation Dropout prevalence: "
        f"{prevalence:.4f}"
    )

    print(
        "Reference Average Precision (random ranking): "
        f"{prevalence:.4f}"
    )

    # ======================================
    # RUN SIX EXPERIMENTS
    # ======================================

    results = []

    models = [
        "logistic",
        "random_forest",
        "xgboost",
    ]

    for stage in ["A", "B"]:

        for model_name in models:

            result = train_binary_model(
                model_name=model_name,
                stage=stage,
                train_df=train_df,
                val_df=val_df,
                config=config,
            )

            results.append(result)

    # ======================================
    # SAVE RESULTS
    # ======================================

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    output_path = (
        REPORTS_DIR /
        "binary_baseline_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================
    # DISPLAY COMPARISON
    # ======================================

    print("\n====================================")
    print("BINARY MODEL COMPARISON")
    print("====================================")

    columns = [
        "model",
        "stage",
        "accuracy",
        "dropout_precision",
        "dropout_recall",
        "dropout_f1",
        "pr_auc_ap",
        "roc_auc",
    ]

    comparison = results_df[
        columns
    ].sort_values(
        ["stage", "pr_auc_ap"],
        ascending=[True, False],
    )

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    print("\n[SAVED REPORT]", output_path)

    print(
        "\n[SUCCESS] "
        "Binary classification training completed."
    )


if __name__ == "__main__":
    main()
