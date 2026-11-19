
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    load_config,
    get_features,
)


# ============================================
# PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"

MANIFEST_PATH = (
    ROOT / "config" / "selected_models.json"
)

VALIDATION_PATH = (
    ROOT / "data" / "splits" / "validation.csv"
)


# ============================================
# FROZEN MODEL FILES
# ============================================

MODEL_FILES = {
    "multiclass_A": "random_forest_A.joblib",

    "multiclass_B":
        "multiclass_logistic_B_balanced.joblib",

    "binary_warning_B":
        "binary_logistic_B_threshold.joblib",
}

CLASSES = [
    "Dropout",
    "Enrolled",
    "Graduate",
]


# ============================================
# PREDICTION ENGINE
# ============================================

class PredictionEngine:

    def __init__(self):

        print("[ENGINE] Initializing AcademicRisk AI...")

        self.config = load_config()

        self.features_A = list(
            get_features(self.config, "A")
        )

        self.features_B = list(
            get_features(self.config, "B")
        )

        self._verify_manifest()

        self.models = {}

        for name, filename in MODEL_FILES.items():

            path = MODEL_DIR / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model: {path}"
                )

            artifact = joblib.load(path)

            if isinstance(artifact, dict):

                if "pipeline" not in artifact:
                    raise ValueError(
                        f"Missing pipeline in {filename}"
                    )

                pipeline = artifact["pipeline"]

            else:
                pipeline = artifact

            self.models[name] = {
                "pipeline": pipeline,
                "artifact": artifact,
            }

            print(f"[LOADED] {filename}")

        binary_artifact = self.models[
            "binary_warning_B"
        ]["artifact"]

        if not isinstance(binary_artifact, dict):
            raise TypeError(
                "Binary model must contain metadata."
            )

        self.threshold = float(
            binary_artifact["threshold"]
        )

        if not np.isclose(
            self.threshold,
            0.22,
        ):
            raise ValueError(
                "Frozen binary threshold mismatch."
            )

        if list(binary_artifact["features"]) != (
            self.features_B
        ):
            raise ValueError(
                "Binary feature order mismatch."
            )

        self._verify_model_features()

        print("[ENGINE] Ready.")
        print("[ENGINE] Threshold:", self.threshold)

    # ----------------------------------------
    # Verify frozen candidate manifest
    # ----------------------------------------

    def _verify_manifest(self):

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Missing manifest: {MANIFEST_PATH}"
            )

        with open(
            MANIFEST_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

        if manifest.get("status") != (
            "frozen_candidates_before_test"
        ):
            raise ValueError(
                "Model manifest is not frozen."
            )

        candidates = manifest.get("candidates", {})

        if set(candidates) != set(MODEL_FILES):
            raise ValueError(
                "Unexpected frozen candidate names."
            )

        for name, filename in MODEL_FILES.items():

            recorded = candidates[name]["file"]

            if Path(recorded).name != filename:
                raise ValueError(
                    f"Manifest mismatch for {name}"
                )

        recorded_threshold = candidates[
            "binary_warning_B"
        ]["threshold"]

        if not np.isclose(
            float(recorded_threshold),
            0.22,
        ):
            raise ValueError(
                "Manifest threshold mismatch."
            )

    # ----------------------------------------
    # Verify input feature order
    # ----------------------------------------

    def _verify_model_features(self):

        for name, stage in [
            ("multiclass_A", "A"),
            ("multiclass_B", "B"),
            ("binary_warning_B", "B"),
        ]:

            pipeline = self.models[name]["pipeline"]

            expected = (
                self.features_A
                if stage == "A"
                else self.features_B
            )

            if not hasattr(
                pipeline,
                "feature_names_in_",
            ):
                raise ValueError(
                    f"Model feature metadata missing: {name}"
                )

            actual = list(
                pipeline.feature_names_in_
            )

            if actual != expected:
                raise ValueError(
                    f"Feature mismatch for {name}"
                )

    # ----------------------------------------
    # Stage selection
    # ----------------------------------------

    def get_required_features(self, stage):

        stage = str(stage).upper()

        if stage == "A":
            return self.features_A.copy()

        if stage == "B":
            return self.features_B.copy()

        raise ValueError(
            "Stage must be A or B."
        )

    # ----------------------------------------
    # Input validation
    # ----------------------------------------

    def validate_input(self, data, stage):

        features = self.get_required_features(stage)

        if isinstance(data, dict):
            df = pd.DataFrame([data])

        elif isinstance(data, pd.DataFrame):
            df = data.copy()

        else:
            raise TypeError(
                "Input must be a dict or DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Input contains no students."
            )

        missing_columns = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_columns)
            )

        # Ignore additional columns such as
        # Target and row_id.
        df = df[features].copy()

        # All 14 original UCI input features
        # are numerically coded.
        # Invalid text is rejected.

        for column in features:

            try:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="raise",
                )

            except (ValueError, TypeError):

                raise ValueError(
                    f"Invalid numeric value in: {column}"
                )

        if df.isna().any().any():

            bad_columns = df.columns[
                df.isna().any()
            ].tolist()

            raise ValueError(
                "Missing values in: "
                + ", ".join(bad_columns)
            )

        values = df.to_numpy(dtype=float)

        if not np.isfinite(values).all():

            raise ValueError(
                "Input contains infinite values."
            )

        # Basic semantic checks

        if "Age at enrollment" in df:

            if (
                df["Age at enrollment"] <= 0
            ).any():

                raise ValueError(
                    "Age must be positive."
                )

        for column in [
            "Admission grade",
            "Previous qualification (grade)",
        ]:

            if column in df:

                if (
                    (df[column] < 0)
                    | (df[column] > 200)
                ).any():

                    raise ValueError(
                        f"{column} must be between 0 and 200."
                    )

        grade_column = (
            "Curricular units 1st sem (grade)"
        )

        if grade_column in df:

            if (
                (df[grade_column] < 0)
                | (df[grade_column] > 20)
            ).any():

                raise ValueError(
                    "First semester grade must "
                    "be between 0 and 20."
                )

        count_columns = [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (without evaluations)",
        ]

        for column in count_columns:

            if column in df:

                if (df[column] < 0).any():
                    raise ValueError(
                        f"{column} cannot be negative."
                    )

                if not np.all(
                    np.isclose(
                        df[column],
                        np.round(df[column]),
                    )
                ):
                    raise ValueError(
                        f"{column} must be an integer."
                    )

        if (
            "Curricular units 1st sem (approved)"
            in df
            and "Curricular units 1st sem (enrolled)"
            in df
        ):

            approved = df[
                "Curricular units 1st sem (approved)"
            ]

            enrolled = df[
                "Curricular units 1st sem (enrolled)"
            ]

            if (approved > enrolled).any():

                raise ValueError(
                    "Approved units cannot exceed "
                    "enrolled units."
                )

        return df

    # ----------------------------------------
    # Multiclass prediction
    # ----------------------------------------

    def _predict_multiclass(self, df, stage):

        model_name = f"multiclass_{stage}"

        pipeline = self.models[
            model_name
        ]["pipeline"]

        predicted = pipeline.predict(df)

        probabilities = pipeline.predict_proba(df)

        classes = list(pipeline.classes_)

        results = []

        for i, prediction in enumerate(predicted):

            class_probabilities = {
                label: float(
                    probabilities[
                        i,
                        classes.index(label)
                    ]
                )
                for label in CLASSES
            }

            results.append({
                "predicted_status": str(prediction),
                "status_probabilities":
                    class_probabilities,
            })

        return results

    # ----------------------------------------
    # Binary early warning
    # ----------------------------------------

    def _predict_binary(self, df):

        pipeline = self.models[
            "binary_warning_B"
        ]["pipeline"]

        classes = list(pipeline.classes_)

        positive_index = classes.index(1)

        probabilities = pipeline.predict_proba(
            df
        )[:, positive_index]

        results = []

        for probability in probabilities:

            probability = float(probability)

            results.append({
                "dropout_risk_score": probability,
                "threshold": self.threshold,
                "warning": bool(
                    probability >= self.threshold
                ),
            })

        return results

    # ----------------------------------------
    # Public prediction interface
    # ----------------------------------------

    def predict(self, data, stage="B"):

        stage = str(stage).upper()

        df = self.validate_input(
            data,
            stage,
        )

        multiclass_results = (
            self._predict_multiclass(
                df,
                stage,
            )
        )

        binary_results = None

        if stage == "B":

            binary_results = (
                self._predict_binary(df)
            )

        results = []

        for i, multiclass_result in enumerate(
            multiclass_results
        ):

            result = {
                "stage": stage,
                **multiclass_result,
            }

            if binary_results is not None:

                result.update(
                    binary_results[i]
                )

            else:

                result.update({
                    "dropout_risk_score": None,
                    "threshold": None,
                    "warning": None,
                })

            results.append(result)

        return results

    # ----------------------------------------
    # Single student prediction
    # ----------------------------------------

    def predict_one(self, student, stage="B"):

        if not isinstance(student, dict):
            raise TypeError(
                "Single student must be a dict."
            )

        return self.predict(
            student,
            stage,
        )[0]


# ============================================
# SMOKE TEST (VALIDATION ONLY)
# ============================================

def run_smoke_test():

    print("\n====================================")
    print("ACADEMICRISK AI - ENGINE TEST")
    print("====================================")

    engine = PredictionEngine()

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    if validation_df.empty:
        raise ValueError(
            "Validation data is empty."
        )

    student = validation_df.iloc[0].to_dict()

    print("\n[TEST 1] Stage A...")

    result_A = engine.predict_one(
        student,
        stage="A",
    )

    print(
        json.dumps(
            result_A,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert result_A["dropout_risk_score"] is None

    print("[PASSED] Stage A")

    print("\n[TEST 2] Stage B...")

    result_B = engine.predict_one(
        student,
        stage="B",
    )

    print(
        json.dumps(
            result_B,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert 0 <= (
        result_B["dropout_risk_score"]
    ) <= 1

    assert np.isclose(
        sum(
            result_B[
                "status_probabilities"
            ].values()
        ),
        1.0,
    )

    # Compare against the original
    # saved model for the same record.

    features = engine.get_required_features("B")

    X = validation_df.iloc[[0]][features]

    original_pipeline = engine.models[
        "binary_warning_B"
    ]["pipeline"]

    original_probability = float(
        original_pipeline.predict_proba(X)[0, 1]
    )

    engine_probability = result_B[
        "dropout_risk_score"
    ]

    if not np.isclose(
        original_probability,
        engine_probability,
        atol=1e-10,
    ):
        raise ValueError(
            "Engine and original model disagree."
        )

    print("[PASSED] Stage B")
    print("[VERIFIED] Prediction matches frozen model.")

    print("\n[TEST 3] Batch prediction...")

    batch = validation_df.head(5)

    batch_results = engine.predict(
        batch,
        stage="B",
    )

    assert len(batch_results) == 5

    print(
        f"[PASSED] Predicted {len(batch_results)} students."
    )

    print("\n[TEST 4] Invalid input...")

    invalid_student = student.copy()

    invalid_student.pop(
        engine.features_B[0],
        None,
    )

    try:

        engine.predict_one(
            invalid_student,
            stage="B",
        )

    except ValueError:

        print(
            "[PASSED] Missing feature rejected."
        )

    else:

        raise AssertionError(
            "Invalid input was not rejected."
        )

    print("\n====================================")
    print("PREDICTION ENGINE TEST COMPLETED")
    print("====================================")

    print("[SUCCESS] All smoke tests passed.")
    print("[INFO] Validation data used.")
    print("[INFO] Final Test not used.")
    print("[INFO] No model retraining.")


if __name__ == "__main__":
    run_smoke_test()
