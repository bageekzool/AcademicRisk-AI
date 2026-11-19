
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
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

    print(f"\n========== MODEL {stage} ==========")

    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    preprocessor = build_preprocessor(
        config,
        features,
        model_type="linear",
    )

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

    custom_weights = {
        "Dropout": 1.5,
        "Enrolled": 2.0,
        "Graduate": 1.0,
    }

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [
            None,
            "balanced",
            custom_weights,
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    print("[INFO] Running 5-fold GridSearchCV...")

    search.fit(X_train, y_train)

    print("\nBest CV Macro F1:", round(
        search.best_score_, 4
    ))

    print("Best parameters:", search.best_params_)

    # The best pipeline is refitted on all training data.
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
        "best_C": search.best_params_[
            "classifier__C"
        ],
        "best_weights": str(
            search.best_params_[
                "classifier__class_weight"
            ]
        ),
    }

    print("\nValidation Macro F1:",
          round(metrics["validation_macro_f1"], 4))
    print("Validation Accuracy:",
          round(metrics["validation_accuracy"], 4))
    print("Dropout Recall:",
          round(metrics["dropout_recall"], 4))
    print("Enrolled Recall:",
          round(metrics["enrolled_recall"], 4))

    # Save CV search results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_)

    cv_results.to_csv(
        REPORTS_DIR / f"logistic_cv_{stage}.csv",
        index=False,
    )

    model_path = (
        MODELS_DIR / f"logistic_cv_{stage}.joblib"
    )

    joblib.dump(best_pipeline, model_path)

    print("[SAVED MODEL]", model_path)

    return metrics


def main():

    print("\n====================================")
    print("ACADEMICRISK AI - LOGISTIC CV TUNING")
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
        result = tune_stage(
            stage,
            train_df,
            val_df,
            config,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path = (
        REPORTS_DIR / "logistic_cv_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== FINAL SUMMARY ==========")
    print(results_df.to_string(index=False))

    print("\n[SAVED REPORT]", output_path)
    print("\n[SUCCESS] CV tuning completed.")


if __name__ == "__main__":
    main()
