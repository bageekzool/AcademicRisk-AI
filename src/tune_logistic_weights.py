
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    log_loss,
    classification_report,
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
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

CLASSES = ["Dropout", "Enrolled", "Graduate"]


# ==========================================
# CLASS WEIGHT EXPERIMENTS
# ==========================================

WEIGHT_CONFIGS = {
    "baseline": None,
    "balanced": "balanced",
    "custom": {
        "Dropout": 1.5,
        "Enrolled": 2.0,
        "Graduate": 1.0,
    },
}


def run_experiment(
    stage,
    weight_name,
    class_weight,
    train_df,
    val_df,
    config,
):

    features = get_features(config, stage)
    target = config["target"]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    preprocessor = build_preprocessor(
        config=config,
        features=features,
        model_type="linear",
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight=class_weight,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)

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

    recalls = recall_score(
        y_val,
        predictions,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    dropout_recall = recalls[0]
    enrolled_recall = recalls[1]
    graduate_recall = recalls[2]

    loss = log_loss(
        y_val,
        probabilities,
        labels=pipeline.named_steps["classifier"].classes_,
    )

    print(
        f"{stage:<5} "
        f"{weight_name:<10} "
        f"{accuracy:.4f} "
        f"{macro_f1:.4f} "
        f"{dropout_recall:.4f} "
        f"{enrolled_recall:.4f} "
        f"{graduate_recall:.4f}"
    )

    return {
        "stage": stage,
        "weight_config": weight_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "dropout_recall": round(dropout_recall, 4),
        "enrolled_recall": round(enrolled_recall, 4),
        "graduate_recall": round(graduate_recall, 4),
        "log_loss": round(loss, 4),
    }


def main():

    print("\n====================================")
    print("ACADEMICRISK AI - CLASS WEIGHT TUNING")
    print("====================================")

    config = load_config()

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        SPLITS_DIR / "validation.csv"
    )

    print("\nClass distribution in training:")
    print(train_df[config["target"]].value_counts())

    print("\n========== EXPERIMENT RESULTS ==========")

    print(
        f"{'Stage':<5} "
        f"{'Weights':<10} "
        f"{'Accuracy':<8} "
        f"{'MacroF1':<8} "
        f"{'Dropout':<8} "
        f"{'Enrolled':<8} "
        f"{'Graduate':<8}"
    )

    results = []

    for stage in ["A", "B"]:

        for weight_name, class_weight in WEIGHT_CONFIGS.items():

            result = run_experiment(
                stage=stage,
                weight_name=weight_name,
                class_weight=class_weight,
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
        REPORTS_DIR / "logistic_class_weight_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== BEST BY STAGE ==========")

    for stage in ["A", "B"]:

        stage_results = (
            results_df[results_df["stage"] == stage]
            .sort_values(
                "macro_f1",
                ascending=False,
            )
        )

        best = stage_results.iloc[0]

        print(
            f"Model {stage}: "
            f"{best['weight_config']} | "
            f"Macro F1 = {best['macro_f1']:.4f} | "
            f"Dropout Recall = "
            f"{best['dropout_recall']:.4f}"
        )

    print("\n[SAVED REPORT]", output_path)

    print("\n[SUCCESS] Weight tuning completed.")


if __name__ == "__main__":
    main()
