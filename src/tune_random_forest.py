
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    log_loss,
)

from src.preprocessing import (
    load_config,
    get_features,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

CLASSES = ["Dropout", "Enrolled", "Graduate"]


def tune_stage(stage, train_df, val_df, config):

    print(f"\n========== RANDOM FOREST {stage} ==========")

    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    preprocessor = build_preprocessor(
        config,
        features,
        model_type="tree",
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
        ),
    ])

    param_distributions = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 8, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.7],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    print("[INFO] Running RandomizedSearchCV...")

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(X_val)
    probabilities = best_pipeline.predict_proba(X_val)

    recalls = recall_score(
        y_val,
        predictions,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    metrics = {
        "stage": stage,
        "cv_macro_f1": search.best_score_,
        "validation_accuracy": accuracy_score(
            y_val, predictions
        ),
        "validation_macro_f1": f1_score(
            y_val,
            predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "dropout_recall": recalls[0],
        "enrolled_recall": recalls[1],
        "graduate_recall": recalls[2],
        "log_loss": log_loss(
            y_val,
            probabilities,
            labels=best_pipeline.named_steps[
                "classifier"
            ].classes_,
        ),
        "best_params": str(search.best_params_),
    }

    print("\nBest CV Macro F1:",
          round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

    print("\n========== VALIDATION ==========")
    print("Accuracy:",
          round(metrics["validation_accuracy"], 4))
    print("Macro F1:",
          round(metrics["validation_macro_f1"], 4))
    print("Dropout Recall:",
          round(metrics["dropout_recall"], 4))
    print("Enrolled Recall:",
          round(metrics["enrolled_recall"], 4))
    print("Log Loss:",
          round(metrics["log_loss"], 4))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv_path = REPORTS_DIR / f"random_forest_cv_{stage}.csv"

    pd.DataFrame(search.cv_results_).to_csv(
        cv_path,
        index=False,
    )

    model_path = MODELS_DIR / f"random_forest_cv_{stage}.joblib"

    joblib.dump(best_pipeline, model_path)

    print("\n[SAVED MODEL]", model_path)
    print("[SAVED CV RESULTS]", cv_path)

    return metrics


def main():

    print("\n====================================")
    print("ACADEMICRISK AI - RANDOM FOREST TUNING")
    print("====================================")

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    results = []

    for stage in ["A", "B"]:
        results.append(
            tune_stage(stage, train_df, val_df, config)
        )

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORTS_DIR / "random_forest_cv_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== FINAL SUMMARY ==========")
    print(results_df.to_string(index=False))

    print("\n[SAVED REPORT]", output_path)
    print("[SUCCESS] Random Forest tuning completed.")


if __name__ == "__main__":
    main()
