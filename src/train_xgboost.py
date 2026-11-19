
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    log_loss,
)

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
# TRAIN XGBOOST
# ==========================================

def train_xgboost(stage, train_df, val_df, config):

    print(f"\n========== XGBOOST {stage} ==========")

    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = train_df[target]
    y_val = val_df[target]

    # Convert class labels into numerical values
    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    y_train_encoded = encoder.transform(y_train)

    # Build the preprocessing pipeline
    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="tree",
    )

    # Initial baseline configuration
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    print("[INFO] Training XGBoost...")

    # Fit only on training records
    pipeline.fit(X_train, y_train_encoded)

    # Predict on validation records
    predictions_encoded = pipeline.predict(X_val)

    probabilities = pipeline.predict_proba(X_val)

    # Restore original class labels
    predictions = encoder.inverse_transform(
        predictions_encoded.astype(int)
    )

    # Evaluation metrics
    accuracy = accuracy_score(
        y_val,
        predictions,
    )

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

    # Probability columns follow the encoded class order
    loss = log_loss(
        encoder.transform(y_val),
        probabilities,
        labels=[0, 1, 2],
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

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    print("Rows = Actual, Columns = Predicted")
    print(matrix_df.to_string())

    # Save model and label encoder together
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODELS_DIR / f"xgboost_{stage}.joblib"

    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": encoder,
            "features": features,
            "stage": stage,
        },
        model_path,
    )

    print(f"\n[SAVED MODEL] {model_path}")

    return {
        "model": f"XGBoost {stage}",
        "stage": stage,
        "original_features": len(features),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "dropout_recall": round(dropout_recall, 4),
        "log_loss": round(loss, 4),
    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - XGBOOST TRAINING")
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

        result = train_xgboost(
            stage=stage,
            train_df=train_df,
            val_df=val_df,
            config=config,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR / "xgboost_validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    f1_a = results_df.loc[
        results_df["stage"] == "A",
        "macro_f1",
    ].iloc[0]

    f1_b = results_df.loc[
        results_df["stage"] == "B",
        "macro_f1",
    ].iloc[0]

    print("\n========== MACRO F1 DIFFERENCE ==========")

    print(f"Model A Macro F1: {f1_a:.4f}")
    print(f"Model B Macro F1: {f1_b:.4f}")
    print(f"Difference (B - A): {f1_b - f1_a:+.4f}")

    print("\n[SAVED REPORT]", output_path)

    print("\n[SUCCESS] XGBoost training completed.")


if __name__ == "__main__":
    main()
