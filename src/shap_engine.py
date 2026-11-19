
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.special import expit

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator


# ============================================
# PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = (
    ROOT / "models" / "binary_B_shap_reference.joblib"
)

TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"


# ============================================
# TURKISH FEATURE NAMES
# ============================================

FEATURE_LABELS = {
    "Application mode": "Başvuru Şekli",
    "Application order": "Başvuru Tercih Sırası",
    "Course": "Bölüm / Program",
    "Daytime/evening attendance": "Öğretim Türü",
    "Previous qualification": "Önceki Eğitim Niteliği",
    "Previous qualification (grade)": "Önceki Eğitim Notu",
    "Admission grade": "Üniversiteye Giriş Notu",
    "Age at enrollment": "Kayıt Yaşı",

    "Curricular units 1st sem (credited)":
        "1. Dönem Sayılan Ders Sayısı",

    "Curricular units 1st sem (enrolled)":
        "1. Dönem Kayıtlı Ders Sayısı",

    "Curricular units 1st sem (evaluations)":
        "1. Dönem Değerlendirme Sayısı",

    "Curricular units 1st sem (approved)":
        "1. Dönem Başarılı Ders Sayısı",

    "Curricular units 1st sem (grade)":
        "1. Dönem Not Ortalaması",

    "Curricular units 1st sem (without evaluations)":
        "1. Dönem Değerlendirmesiz Ders Sayısı",
}


# ============================================
# HELPER
# ============================================

def to_dense(data):

    if hasattr(data, "toarray"):
        return data.toarray()

    return np.asarray(data)


# ============================================
# SHAP ENGINE
# ============================================

class ShapEngine:

    def __init__(self, prediction_engine=None):

        print("[SHAP ENGINE] Initializing...")

        if prediction_engine is None:
            prediction_engine = PredictionEngine()

        self.engine = prediction_engine

        if not REFERENCE_PATH.exists():
            raise FileNotFoundError(
                f"Missing SHAP reference: {REFERENCE_PATH}"
            )

        self.reference = joblib.load(REFERENCE_PATH)

        artifact = self.engine.models[
            "binary_warning_B"
        ]["artifact"]

        self.pipeline = artifact["pipeline"]

        self.preprocessor = self.pipeline.named_steps[
            "preprocessing"
        ]

        self.classifier = self.pipeline.named_steps[
            "classifier"
        ]

        self.features = list(
            artifact["features"]
        )

        self.threshold = float(
            artifact["threshold"]
        )

        # ------------------------------------
        # Validate reference metadata
        # ------------------------------------

        if self.reference["model"] != "binary_warning_B":
            raise ValueError(
                "SHAP reference model mismatch."
            )

        if list(self.reference["features"]) != self.features:
            raise ValueError(
                "SHAP reference feature mismatch."
            )

        if not np.isclose(
            float(self.reference["threshold"]),
            self.threshold,
        ):
            raise ValueError(
                "SHAP threshold mismatch."
            )

        if list(self.classifier.classes_) != [0, 1]:
            raise ValueError(
                "Unexpected binary class order."
            )

        self.transformed_names = list(
            self.preprocessor.get_feature_names_out()
        )

        saved_names = list(
            self.reference["transformed_feature_names"]
        )

        if self.transformed_names != saved_names:
            raise ValueError(
                "Transformed feature names mismatch."
            )

        self.reference_mean = np.asarray(
            self.reference["reference_mean"],
            dtype=float,
        )

        self.coefficients = np.asarray(
            self.classifier.coef_[0],
            dtype=float,
        )

        self.base_log_odds = float(
            self.reference["base_log_odds"]
        )

        self.intercept = float(
            self.classifier.intercept_[0]
        )

        count = len(self.transformed_names)

        if (
            len(self.reference_mean) != count
            or len(self.coefficients) != count
        ):
            raise ValueError(
                "Reference dimensions mismatch."
            )

        expected_base = (
            self.intercept
            + float(
                np.dot(
                    self.coefficients,
                    self.reference_mean,
                )
            )
        )

        if not np.isclose(
            expected_base,
            self.base_log_odds,
            atol=1e-8,
        ):
            raise ValueError(
                "SHAP base value mismatch."
            )

        # Match each transformed feature
        # to its original input feature.
        self.source_features = [
            self._resolve_source(name)
            for name in self.transformed_names
        ]

        print("[SHAP ENGINE] Ready.")
        print(
            "[SHAP ENGINE] Original features:",
            len(self.features),
        )
        print(
            "[SHAP ENGINE] Transformed features:",
            count,
        )

    # ----------------------------------------
    # Map transformed feature to source
    # ----------------------------------------
    def _resolve_source(self, transformed_name):

        # Numerical features
        if transformed_name.startswith("numeric__"):

            source = transformed_name[len("numeric__"):]

            if source not in self.features:
                raise ValueError(
                    f"Unknown numeric feature: {source}"
                )

            return source

        # Binary features
        if transformed_name.startswith("binary__"):

            source = transformed_name[len("binary__"):]

            if source not in self.features:
                raise ValueError(
                    f"Unknown binary feature: {source}"
                )

            return source

        # One-hot encoded categorical features
        if transformed_name.startswith("categorical__"):

            encoded_name = transformed_name[
                len("categorical__"):
            ]

            for source in sorted(
                self.features,
                key=len,
                reverse=True,
            ):

                if encoded_name.startswith(source + "_"):
                    return source

            raise ValueError(
                f"Unknown categorical feature: "
                f"{transformed_name}"
            )

        # Direct passthrough features
        if transformed_name in self.features:
            return transformed_name

        raise ValueError(
            f"Cannot map feature: {transformed_name}"
        )

    # ----------------------------------------
    # Explain one student
    # ----------------------------------------

    def explain_one(self, student):

        # Do not bypass input validation.
        validator = StudentInputValidator(
            self.engine
        )

        df = validator.validate(
            student,
            stage="B",
        )

        if len(df) != 1:
            raise ValueError(
                "explain_one requires one student."
            )

        transformed = np.asarray(
            to_dense(
                self.preprocessor.transform(df)
            ),
            dtype=float,
        )

        x = transformed[0]

        # Exact linear SHAP contributions
        # relative to the saved background mean.
        shap_values = (
            x - self.reference_mean
        ) * self.coefficients

        total_log_odds = (
            self.base_log_odds
            + float(shap_values.sum())
        )

        reconstructed_probability = float(
            expit(total_log_odds)
        )

        predicted_probability = float(
            self.pipeline.predict_proba(df)[0, 1]
        )

        difference = abs(
            reconstructed_probability
            - predicted_probability
        )

        if difference > 1e-8:
            raise ValueError(
                "SHAP explanation does not "
                "reproduce the model probability."
            )

        # ------------------------------------
        # Aggregate encoded columns into
        # the original 14 input features.
        # ------------------------------------

        contributions = {
            feature: 0.0
            for feature in self.features
        }

        for source, value in zip(
            self.source_features,
            shap_values,
        ):

            contributions[source] += float(value)

        rows = []

        original_row = df.iloc[0]

        for feature in self.features:

            contribution = contributions[feature]

            original_value = original_row[feature]

            rows.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(
                    feature,
                    feature,
                ),
                "original_value": float(
                    original_value
                ),
                "shap_log_odds": contribution,
                "abs_shap": abs(contribution),
                "direction": (
                    "Artırıyor"
                    if contribution > 0
                    else (
                        "Azaltıyor"
                        if contribution < 0
                        else "Nötr"
                    )
                ),
            })

        contributions_df = pd.DataFrame(
            rows
        ).sort_values(
            "abs_shap",
            ascending=False,
        ).reset_index(drop=True)

        aggregated_total = float(
            contributions_df[
                "shap_log_odds"
            ].sum()
        )

        if not np.isclose(
            aggregated_total,
            shap_values.sum(),
            atol=1e-8,
        ):
            raise ValueError(
                "Aggregated contributions mismatch."
            )

        return {
            "dropout_probability": predicted_probability,
            "threshold": self.threshold,
            "warning": bool(
                predicted_probability >= self.threshold
            ),
            "base_log_odds": self.base_log_odds,
            "base_probability": float(
                expit(self.base_log_odds)
            ),
            "final_log_odds": float(total_log_odds),
            "reconstruction_error": float(difference),
            "contributions": contributions_df,
        }


# ============================================
# TEST
# ============================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - SHAP ENGINE TEST")
    print("====================================")

    prediction_engine = PredictionEngine()

    shap_engine = ShapEngine(
        prediction_engine
    )

    # Use TRAINING data for the smoke test.
    # Do not reopen the final Test set.

    train_df = pd.read_csv(TRAIN_PATH)

    student = train_df.iloc[0].to_dict()

    print("\n[STEP 1] Explaining student...")

    result = shap_engine.explain_one(
        student
    )

    print(
        "Dropout probability:",
        f"{result['dropout_probability']:.6f}",
    )

    print(
        "Base probability:",
        f"{result['base_probability']:.6f}",
    )

    print(
        "Warning:",
        result["warning"],
    )

    print(
        "Reconstruction error:",
        f"{result['reconstruction_error']:.10f}",
    )

    print("\n[TOP 10 CONTRIBUTIONS]")

    columns = [
        "label",
        "original_value",
        "shap_log_odds",
        "direction",
    ]

    print(
        result["contributions"][
            columns
        ].head(10).to_string(
            index=False
        )
    )

    assert len(
        result["contributions"]
    ) == 14

    assert result[
        "reconstruction_error"
    ] < 1e-8

    print("\n====================================")
    print("SHAP ENGINE TEST COMPLETED")
    print("====================================")

    print("[SUCCESS] SHAP engine works.")
    print("[SUCCESS] 63 features grouped into 14.")
    print("[SUCCESS] Probability verified.")
    print("[INFO] Frozen models unchanged.")
    print("[INFO] Final Test not accessed.")


if __name__ == "__main__":
    main()
