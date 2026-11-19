
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.prediction_engine import PredictionEngine


# ============================================
# PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"


# ============================================
# VALIDATION RULES
# ============================================

BINARY_COLUMNS = {
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
}

INTEGER_COLUMNS = {
    "Age at enrollment",
    "Application order",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
}

GRADE_RANGES = {
    "Admission grade": (0, 200),
    "Previous qualification (grade)": (0, 200),
    "Curricular units 1st sem (grade)": (0, 20),
}


# ============================================
# FIND FITTED ONE-HOT ENCODER
# ============================================

def find_onehot_encoder(transformer):

    if isinstance(transformer, OneHotEncoder):
        return transformer

    if hasattr(transformer, "named_steps"):

        for step in transformer.named_steps.values():

            result = find_onehot_encoder(step)

            if result is not None:
                return result

    return None


# ============================================
# STUDENT INPUT VALIDATOR
# ============================================

class StudentInputValidator:

    def __init__(self, engine):

        self.engine = engine

        # Categories are obtained from the
        # already-fitted model preprocessing.
        # No training dataset is needed at
        # prediction time.

        self.allowed_categories = {
            "A": self._get_categories("A"),
            "B": self._get_categories("B"),
        }

        print("[VALIDATOR] Initialized.")

    def _get_categories(self, stage):

        model_name = f"multiclass_{stage}"

        pipeline = self.engine.models[
            model_name
        ]["pipeline"]

        preprocessor = pipeline.named_steps[
            "preprocessing"
        ]

        allowed = {}

        for name, transformer, columns in (
            preprocessor.transformers_
        ):

            encoder = find_onehot_encoder(
                transformer
            )

            if encoder is None:
                continue

            if not hasattr(encoder, "categories_"):
                raise ValueError(
                    "OneHotEncoder is not fitted."
                )

            column_names = list(columns)

            if len(column_names) != len(
                encoder.categories_
            ):
                raise ValueError(
                    "Categorical schema mismatch."
                )

            for column, categories in zip(
                column_names,
                encoder.categories_,
            ):

                allowed[column] = list(
                    categories
                )

        return allowed

    def get_allowed_categories(self, stage="B"):

        stage = stage.upper()

        if stage not in self.allowed_categories:
            raise ValueError(
                "Stage must be A or B."
            )

        return {
            column: list(values)
            for column, values in
            self.allowed_categories[stage].items()
        }

    def validate(self, data, stage="B"):

        stage = stage.upper()

        # The existing engine already checks
        # required columns, missing values,
        # numeric values and basic constraints.

        df = self.engine.validate_input(
            data,
            stage,
        )

        errors = []

        # ------------------------------------
        # 1. Check allowed category codes
        # ------------------------------------

        for column, categories in (
            self.allowed_categories[stage].items()
        ):

            if column not in df.columns:
                continue

            allowed = set(
                float(value)
                for value in categories
            )

            invalid = [
                value
                for value in df[column].unique()
                if float(value) not in allowed
            ]

            if invalid:

                errors.append(
                    f"{column}: eğitim sırasında "
                    f"görülmeyen kategori {invalid}"
                )

        # ------------------------------------
        # 2. Check binary columns
        # ------------------------------------

        for column in (
            BINARY_COLUMNS.intersection(df.columns)
        ):

            invalid_mask = ~df[column].isin(
                [0, 1]
            )

            if invalid_mask.any():

                errors.append(
                    f"{column}: yalnızca 0 "
                    f"veya 1 girilebilir."
                )

        # ------------------------------------
        # 3. Check integer features
        # ------------------------------------

        for column in (
            INTEGER_COLUMNS.intersection(df.columns)
        ):

            values = df[column].to_numpy()

            if not np.all(
                np.isclose(
                    values,
                    np.round(values),
                )
            ):

                errors.append(
                    f"{column}: tam sayı olmalıdır."
                )

        # ------------------------------------
        # 4. Check age range
        # ------------------------------------

        if "Age at enrollment" in df.columns:

            age = df["Age at enrollment"]

            if ((age < 1) | (age > 120)).any():

                errors.append(
                    "Kayıt yaşı 1 ile 120 "
                    "arasında olmalıdır."
                )

        # ------------------------------------
        # 5. Check grades
        # ------------------------------------

        for column, (minimum, maximum) in (
            GRADE_RANGES.items()
        ):

            if column not in df.columns:
                continue

            invalid = (
                (df[column] < minimum)
                | (df[column] > maximum)
            )

            if invalid.any():

                errors.append(
                    f"{column}: {minimum} ile "
                    f"{maximum} arasında olmalıdır."
                )

        # ------------------------------------
        # 6. Check academic counts
        # ------------------------------------

        count_columns = [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (without evaluations)",
        ]

        for column in count_columns:

            if column in df.columns:

                if (df[column] < 0).any():

                    errors.append(
                        f"{column}: negatif olamaz."
                    )

        approved_col = (
            "Curricular units 1st sem (approved)"
        )

        enrolled_col = (
            "Curricular units 1st sem (enrolled)"
        )

        if (
            approved_col in df.columns
            and enrolled_col in df.columns
        ):

            if (
                df[approved_col] > df[enrolled_col]
            ).any():

                errors.append(
                    "Başarılı ders sayısı kayıtlı "
                    "ders sayısından fazla olamaz."
                )

        # ------------------------------------
        # FINAL RESULT
        # ------------------------------------

        if errors:

            raise ValueError(
                "\n".join(errors)
            )

        return df


# ============================================
# TEST HELPERS
# ============================================

def expect_rejected(
    validator,
    student,
    stage,
    label,
):

    try:

        validator.validate(
            student,
            stage=stage,
        )

    except ValueError as error:

        print(f"[PASSED] {label}")
        print(" Reason:", str(error).splitlines()[0])

    else:

        raise AssertionError(
            f"{label}: invalid input accepted."
        )


# ============================================
# SMOKE TEST
# ============================================

def main():

    print("\n====================================")
    print("ACADEMICRISK AI - INPUT VALIDATOR")
    print("====================================")

    engine = PredictionEngine()

    validator = StudentInputValidator(
        engine
    )

    print("\n[STEP 1] Categorical schema")

    for stage in ["A", "B"]:

        allowed = validator.get_allowed_categories(
            stage
        )

        print(f"\nStage {stage}")

        for column, categories in allowed.items():

            print(
                f"  {column}: "
                f"{len(categories)} categories"
            )

    # Test with a TRAINING record.
    # No Test data is accessed.

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    student = train_df.iloc[0].to_dict()

    print("\n[STEP 2] Valid student")

    validated = validator.validate(
        student,
        stage="B",
    )

    print("[PASSED] Valid student accepted.")
    print("Features:", len(validated.columns))

    # ----------------------------------------
    # Missing feature
    # ----------------------------------------

    print("\n[STEP 3] Missing feature")

    invalid = student.copy()

    invalid.pop(
        engine.features_B[0],
        None,
    )

    expect_rejected(
        validator,
        invalid,
        "B",
        "Missing feature",
    )

    # ----------------------------------------
    # Invalid grade
    # ----------------------------------------

    print("\n[STEP 4] Invalid grade")

    invalid = student.copy()

    invalid[
        "Curricular units 1st sem (grade)"
    ] = 85

    expect_rejected(
        validator,
        invalid,
        "B",
        "Invalid semester grade",
    )

    # ----------------------------------------
    # Invalid categorical code
    # ----------------------------------------

    print("\n[STEP 5] Unknown category")

    categorical_schema = (
        validator.get_allowed_categories("B")
    )

    if categorical_schema:

        column = next(
            iter(categorical_schema)
        )

        invalid = student.copy()

        invalid[column] = 999999

        expect_rejected(
            validator,
            invalid,
            "B",
            "Unknown category",
        )

    else:

        raise AssertionError(
            "No fitted categorical schema found."
        )

    # ----------------------------------------
    # Invalid academic counts
    # ----------------------------------------

    print("\n[STEP 6] Academic consistency")

    invalid = student.copy()

    invalid[
        "Curricular units 1st sem (enrolled)"
    ] = 3

    invalid[
        "Curricular units 1st sem (approved)"
    ] = 8

    expect_rejected(
        validator,
        invalid,
        "B",
        "Approved exceeds enrolled",
    )

    print("\n====================================")
    print("INPUT VALIDATION COMPLETED")
    print("====================================")

    print("[SUCCESS] All checks passed.")
    print("[INFO] Frozen models unchanged.")
    print("[INFO] Final Test not accessed.")


if __name__ == "__main__":
    main()
