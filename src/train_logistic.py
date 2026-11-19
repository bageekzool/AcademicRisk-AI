
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

CLASSES = ["Dropout", "Enrolled", "Graduate"]


# ==========================================
# TRAIN ONE LOGISTIC REGRESSION MODEL
# ==========================================

def train_model(stage, train_df, val_df, config):

    print(f"\n========== LOGISTIC REGRESSION {stage} ==========")

    # Select features based on the academic stage
    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    # Build preprocessing for a linear model
    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="linear",
    )

    # Create the complete ML pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    # Fit preprocessing and classifier on TRAIN only
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    # Predict on validation data
    predictions = pipeline.predict(X_val)

    probabilities = pipeline.predict_proba(X_val)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    dropout_recall = recall_score(
        y_val,
        predictions,
        labels=["Dropout"],
        average="macro",
        zero_division=0,
    )

    loss = log_loss(
        y_val,
        probabilities,
        labels=pipeline.named_steps[
            "classifier"
        ].classes_,
    )

    print("\n========== METRICS ==========")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Dropout Recall: {dropout_recall:.4f}")
    print(f"Log Loss:       {loss:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("\n========== CONFUSION MATRIX ==========")
    cm = confusion_matrix(
        y_val,
        predictions,
        labels=CLASSES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    )

    print("Rows = Actual, Columns = Predicted")
    print(cm_df.to_string())

    # Save complete trained pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"logistic_{stage}.joblib"

    joblib.dump(pipeline, model_path)

    print(f"\n[SAVED MODEL] {model_path}")

    return {
        "model": f"Logistic Regression {stage}",
        "stage": stage,
        "original_features": len(features),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "dropout_recall": round(dropout_recall, 4),
        "log_loss": round(loss, 4),
    }


# ==========================================
# MAIN TRAINING WORKFLOW
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - FIRST MODEL TRAINING")
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

    results = []

    for stage in ["A", "B"]:
        result = train_model(
            stage=stage,
            train_df=train_df,
            val_df=val_df,
            config=config,
        )

        results.append(result)

    # Save comparison results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    output_path = (
        REPORTS_DIR / "logistic_validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\n========== MACRO F1 DIFFERENCE ==========")

    f1_a = results_df.loc[
        results_df["stage"] == "A",
        "macro_f1",
    ].iloc[0]

    f1_b = results_df.loc[
        results_df["stage"] == "B",
        "macro_f1",
    ].iloc[0]

    difference = f1_b - f1_a

    print(f"Model A Macro F1: {f1_a:.4f}")
    print(f"Model B Macro F1: {f1_b:.4f}")
    print(f"Difference (B - A): {difference:+.4f}")

    print("\n[SAVED REPORT]", output_path)

    print("\n[SUCCESS] Logistic Regression training completed.")


if __name__ == "__main__":
    main()
