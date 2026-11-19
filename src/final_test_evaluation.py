
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    log_loss,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from src.preprocessing import load_config, get_features
from src.train_binary import make_binary_target


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = ROOT / "data" / "splits" / "test.csv"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "selected_models.json"

TABLES_DIR = ROOT / "reports" / "tables"
FINAL_DIR = ROOT / "reports" / "final_test"

CLASSES = ["Dropout", "Enrolled", "Graduate"]


# ==========================================
# LOAD FROZEN CANDIDATES
# ==========================================

def load_manifest():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("status") != "frozen_candidates_before_test":
        raise ValueError("Candidates are not frozen.")

    return manifest


def load_candidate(metadata):

    model_path = MODELS_DIR / metadata["file"]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        pipeline = artifact["pipeline"]
        saved_features = artifact.get("features")
    else:
        pipeline = artifact
        saved_features = None

    return pipeline, artifact, saved_features


# ==========================================
# MULTICLASS EVALUATION
# ==========================================

def evaluate_multiclass(
    name,
    metadata,
    test_df,
    config,
):

    print(f"\n========== {name.upper()} ==========")

    features = get_features(
        config,
        metadata["stage"],
    )

    pipeline, artifact, saved_features = (
        load_candidate(metadata)
    )

    if saved_features is not None:
        if list(saved_features) != list(features):
            raise ValueError("Feature mismatch.")

    X_test = test_df[features]
    y_test = test_df[config["target"]]

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    model_classes = list(pipeline.classes_)

    # Align probability columns with the fixed class order
    class_indices = [
        model_classes.index(label)
        for label in CLASSES
    ]

    aligned_probs = probabilities[:, class_indices]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    recalls = recall_score(
        y_test,
        predictions,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    loss = log_loss(
        y_test,
        aligned_probs,
        labels=CLASSES,
    )

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Dropout Recall: {recalls[0]:.4f}")
    print(f"Enrolled Recall:{recalls[1]:.4f}")
    print(f"Graduate Recall:{recalls[2]:.4f}")
    print(f"Log Loss:       {loss:.4f}")

    print("\nCLASSIFICATION REPORT:")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0,
    )

    print(report)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    print("\nCONFUSION MATRIX:")
    print("Rows = actual, columns = predicted")
    print(matrix_df.to_string())

    matrix_df.to_csv(
        FINAL_DIR / f"{name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    (FINAL_DIR / f"{name}_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    return {
        "task": "multiclass",
        "model": name,
        "stage": metadata["stage"],
        "test_students": len(y_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "dropout_recall": recalls[0],
        "enrolled_recall": recalls[1],
        "graduate_recall": recalls[2],
        "log_loss": loss,
        "threshold": np.nan,
    }


# ==========================================
# BINARY WARNING EVALUATION
# ==========================================

def evaluate_binary(
    name,
    metadata,
    test_df,
    config,
):

    print(f"\n========== {name.upper()} ==========")

    features = get_features(
        config,
        metadata["stage"],
    )

    pipeline, artifact, saved_features = (
        load_candidate(metadata)
    )

    if not isinstance(artifact, dict):
        raise ValueError(
            "Binary model metadata is missing."
        )

    if list(artifact["features"]) != list(features):
        raise ValueError("Binary feature mismatch.")

    threshold = float(artifact["threshold"])

    if abs(threshold - float(metadata["threshold"])) > 1e-9:
        raise ValueError("Threshold mismatch.")

    X_test = test_df[features]

    y_test = make_binary_target(
        test_df[config["target"]]
    )

    model_classes = list(pipeline.classes_)
    positive_index = model_classes.index(1)

    probabilities = pipeline.predict_proba(X_test)[
        :, positive_index
    ]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    dropout_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_test,
        predictions,
        beta=2,
        zero_division=0,
    )

    ap = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1],
    )

    print(f"Threshold:         {threshold:.2f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Dropout Precision: {precision:.4f}")
    print(f"Dropout Recall:    {recall:.4f}")
    print(f"Dropout F1:        {dropout_f1:.4f}")
    print(f"Dropout F2:        {f2:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Brier Score:       {brier:.4f}")
    print(f"Log Loss:          {loss:.4f}")

    print("\nDROPOUT DETECTION:")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")
    print(f"Total alerts:    {tp + fp}")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Dropout", "Dropout"],
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT:")
    print(report)

    matrix_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Not Dropout", "Dropout"],
        columns=["Not Dropout", "Dropout"],
    )

    matrix_df.to_csv(
        FINAL_DIR / f"{name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    (FINAL_DIR / f"{name}_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    return {
        "task": "binary_warning",
        "model": name,
        "stage": metadata["stage"],
        "test_students": len(y_test),
        "threshold": threshold,
        "accuracy": accuracy,
        "dropout_precision": precision,
        "dropout_recall": recall,
        "dropout_f1": dropout_f1,
        "dropout_f2": f2,
        "average_precision": ap,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "log_loss": loss,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - FINAL TEST")
    print("====================================")

    # Avoid overwriting an existing final report
    output_path = FINAL_DIR / "final_test_results.csv"

    if output_path.exists():
        raise FileExistsError(
            "Final test report already exists. "
            "Do not rerun model selection using Test."
        )

    config = load_config()
    manifest = load_manifest()

    candidates = manifest["candidates"]

    expected = {
        "multiclass_A",
        "multiclass_B",
        "binary_warning_B",
    }

    if set(candidates) != expected:
        raise ValueError("Unexpected candidate selection.")

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    test_df = pd.read_csv(TEST_PATH)

    if len(test_df) != 664:
        raise ValueError(
            f"Unexpected test size: {len(test_df)}"
        )

    print("Test students:", len(test_df))
    print("Frozen models:", len(candidates))

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for name in ["multiclass_A", "multiclass_B"]:

        result = evaluate_multiclass(
            name=name,
            metadata=candidates[name],
            test_df=test_df,
            config=config,
        )

        results.append(result)

    result = evaluate_binary(
        name="binary_warning_B",
        metadata=candidates["binary_warning_B"],
        test_df=test_df,
        config=config,
    )

    results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n====================================")
    print("FINAL TEST SUMMARY")
    print("====================================")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\n[SAVED REPORT]", output_path)

    print(
        "\n[SUCCESS] Final test evaluation completed."
    )

    print(
        "[IMPORTANT] Keep the models and thresholds "
        "unchanged after this evaluation."
    )


if __name__ == "__main__":
    main()
