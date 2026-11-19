
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
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


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

CLASSES = ["Dropout", "Enrolled", "Graduate"]

RANDOM_STATE = 42


# ==========================================
# TUNE ONE ACADEMIC STAGE
# ==========================================

def tune_stage(stage, train_df, val_df, config):

    print(f"\n========== XGBOOST {stage} ==========")

    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    X_val = val_df[features]

    y_train = train_df[target]
    y_val = val_df[target]

    # XGBoost requires numerical class labels
    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="tree",
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ])

    # Hyperparameter search space
    param_distributions = {
        "classifier__n_estimators": [
            150, 250, 350
        ],
        "classifier__max_depth": [
            2, 3, 4, 5
        ],
        "classifier__learning_rate": [
            0.03, 0.05, 0.1
        ],
        "classifier__subsample": [
            0.8, 0.9, 1.0
        ],
        "classifier__colsample_bytree": [
            0.8, 0.9, 1.0
        ],
        "classifier__min_child_weight": [
            1, 3, 5
        ],
        "classifier__reg_lambda": [
            1.0, 5.0, 10.0
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=2,
        refit=True,
        error_score="raise",
        verbose=1,
    )

    print("[INFO] Running XGBoost RandomizedSearchCV...")

    # Only training data enters the search
    search.fit(X_train, y_train_encoded)

    best_pipeline = search.best_estimator_

    # Validation evaluation
    predictions_encoded = (
        best_pipeline.predict(X_val).astype(int)
    )

    predictions = encoder.inverse_transform(
        predictions_encoded
    )

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
            y_val_encoded,
            probabilities,
            labels=[0, 1, 2],
        ),
        "best_params": str(search.best_params_),
    }

    print("\n========== BEST CV RESULTS ==========")
    print(f"CV Macro F1: {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

    print("\n========== VALIDATION RESULTS ==========")

    for metric in [
        "validation_accuracy",
        "validation_macro_f1",
        "dropout_recall",
        "enrolled_recall",
        "graduate_recall",
        "log_loss",
    ]:
        print(f"{metric}: {metrics[metric]:.4f}")

    # Save reports and trained pipeline
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv_path = REPORTS_DIR / f"xgboost_cv_{stage}.csv"

    pd.DataFrame(search.cv_results_).to_csv(
        cv_path,
        index=False,
    )

    model_path = MODELS_DIR / f"xgboost_cv_{stage}.joblib"

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "label_encoder": encoder,
            "features": features,
            "stage": stage,
        },
        model_path,
    )

    print("\n[SAVED MODEL]", model_path)
    print("[SAVED CV RESULTS]", cv_path)

    return metrics


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - XGBOOST CV TUNING")
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
            tune_stage(
                stage,
                train_df,
                val_df,
                config,
            )
        )

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "xgboost_cv_summary.csv"

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== FINAL SUMMARY ==========")
    print(results_df.to_string(index=False))

    print("\n[SAVED REPORT]", output_path)

    print("\n[SUCCESS] XGBoost tuning completed.")


if __name__ == "__main__":
    main()
