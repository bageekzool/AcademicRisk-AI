
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
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
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42

# Training sizes relative to each CV training fold.
TRAIN_SIZES = np.array([
    0.10,
    0.20,
    0.35,
    0.50,
    0.70,
    1.00,
])


# ==========================================
# ANALYZE ONE MODEL STAGE
# ==========================================

def analyze_stage(stage, train_df, config):

    print(f"\n========== LEARNING CURVE - MODEL {stage} ==========")

    features = get_features(config, stage)
    target = config["target"]

    X = train_df[features]
    y = train_df[target]

    print(f"Original features: {len(features)}")
    print(f"Available training records: {len(X)}")

    # All preprocessing is fitted inside each CV fold.
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
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # Internal 5-fold CV using training data only.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("[INFO] Calculating learning curves...")

    sizes, train_scores, cv_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=TRAIN_SIZES,
        cv=cv,
        scoring="f1_macro",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=2,
        error_score="raise",
    )

    # Average and variability across CV folds.
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    cv_mean = cv_scores.mean(axis=1)
    cv_std = cv_scores.std(axis=1)

    results = pd.DataFrame({
        "stage": stage,
        "train_size": sizes,
        "train_f1_mean": train_mean,
        "train_f1_std": train_std,
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "generalization_gap": train_mean - cv_mean,
    })

    print("\n========== RESULTS ==========")

    print(
        results[
            [
                "train_size",
                "train_f1_mean",
                "cv_f1_mean",
                "generalization_gap",
            ]
        ].round(4).to_string(index=False)
    )

    # ======================================
    # SAVE NUMERICAL RESULTS
    # ======================================

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR /
        f"learning_curve_{stage}.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================
    # CREATE LEARNING CURVE FIGURE
    # ======================================

    plt.figure(figsize=(9, 6))

    plt.plot(
        sizes,
        train_mean,
        marker="o",
        linewidth=2,
        label="Training Macro F1",
    )

    plt.plot(
        sizes,
        cv_mean,
        marker="o",
        linewidth=2,
        label="Cross-Validation Macro F1",
    )

    plt.fill_between(
        sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    plt.fill_between(
        sizes,
        cv_mean - cv_std,
        cv_mean + cv_std,
        alpha=0.15,
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Macro F1 Score")

    plt.title(
        f"AcademicRisk AI - Learning Curve (Model {stage})"
    )

    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure_path = (
        FIGURES_DIR /
        f"learning_curve_{stage}.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\n[SAVED TABLE]", csv_path)
    print("[SAVED FIGURE]", figure_path)

    return results


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - LEARNING CURVES")
    print("====================================")

    config = load_config()

    # Do not load validation.csv or test.csv.
    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    print(f"\nTraining records: {len(train_df)}")
    print("Evaluation method: 5-Fold Cross-Validation")
    print("Scoring metric: Macro F1")

    all_results = []

    for stage in ["A", "B"]:

        results = analyze_stage(
            stage=stage,
            train_df=train_df,
            config=config,
        )

        all_results.append(results)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        TABLES_DIR /
        "learning_curves_comparison.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n====================================")
    print("LEARNING CURVE ANALYSIS COMPLETED")
    print("====================================")

    print("[SAVED COMPARISON]", output_path)
    print("[SUCCESS] Learning curves generated.")


if __name__ == "__main__":
    main()
