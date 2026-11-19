
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator


# ============================================
# PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"


# ============================================
# FIXTURES
# ============================================

@pytest.fixture(scope="module")
def engine():

    return PredictionEngine()


@pytest.fixture(scope="module")
def validator(engine):

    return StudentInputValidator(engine)


@pytest.fixture(scope="module")
def training_samples():

    if not TRAIN_PATH.exists():
        pytest.fail(
            f"Training file missing: {TRAIN_PATH}"
        )

    data = pd.read_csv(TRAIN_PATH)

    assert len(data) >= 5

    # Functional tests only.
    # Do not use the Final Test dataset.
    return data.head(5).copy()


@pytest.fixture
def student(training_samples):

    return training_samples.iloc[0].to_dict()


# ============================================
# TEST 01: MODEL LOADING
# ============================================

def test_models_loaded(engine):

    expected_models = [
        "multiclass_A",
        "multiclass_B",
        "binary_warning_B",
    ]

    for name in expected_models:

        assert name in engine.models


# ============================================
# TEST 02: FEATURE COUNTS
# ============================================

def test_feature_counts(engine):

    stage_a = engine.get_required_features("A")
    stage_b = engine.get_required_features("B")

    assert len(stage_a) == 8
    assert len(stage_b) == 14

    assert set(stage_a).issubset(set(stage_b))

    assert len(stage_a) == len(set(stage_a))
    assert len(stage_b) == len(set(stage_b))


# ============================================
# TEST 03: VALID STUDENT
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_valid_student(
    validator,
    student,
    stage,
):

    validated = validator.validate(
        student,
        stage=stage,
    )

    assert len(validated) == 1

    assert not validated.isna().any().any()


# ============================================
# TEST 04: MISSING FEATURE
# ============================================

def test_missing_feature_rejected(
    engine,
    validator,
    student,
):

    invalid = student.copy()

    first_feature = engine.get_required_features(
        "B"
    )[0]

    invalid.pop(first_feature)

    with pytest.raises(ValueError):

        validator.validate(
            invalid,
            stage="B",
        )


# ============================================
# TEST 05: UNKNOWN CATEGORY
# ============================================

def test_unknown_category_rejected(
    validator,
    student,
):

    invalid = student.copy()

    invalid["Course"] = 999999

    with pytest.raises(ValueError):

        validator.validate(
            invalid,
            stage="B",
        )


# ============================================
# TEST 06: INVALID GRADE
# ============================================

@pytest.mark.parametrize(
    "invalid_grade",
    [-1, 21, 100],
)
def test_invalid_semester_grade(
    validator,
    student,
    invalid_grade,
):

    invalid = student.copy()

    invalid[
        "Curricular units 1st sem (grade)"
    ] = invalid_grade

    with pytest.raises(ValueError):

        validator.validate(
            invalid,
            stage="B",
        )


# ============================================
# TEST 07: INVALID AGE
# ============================================

@pytest.mark.parametrize(
    "invalid_age",
    [-5, 0, 150],
)
def test_invalid_age(
    validator,
    student,
    invalid_age,
):

    invalid = student.copy()

    invalid["Age at enrollment"] = invalid_age

    with pytest.raises(ValueError):

        validator.validate(
            invalid,
            stage="B",
        )


# ============================================
# TEST 08: ACADEMIC CONSISTENCY
# ============================================

def test_approved_exceeds_enrolled(
    validator,
    student,
):

    invalid = student.copy()

    invalid[
        "Curricular units 1st sem (enrolled)"
    ] = 3

    invalid[
        "Curricular units 1st sem (approved)"
    ] = 8

    with pytest.raises(ValueError):

        validator.validate(
            invalid,
            stage="B",
        )


# ============================================
# TEST 09: PREDICTION PROBABILITIES
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_prediction_probabilities(
    engine,
    validator,
    student,
    stage,
):

    clean = validator.validate(
        student,
        stage=stage,
    )

    result = engine.predict_one(
        clean.iloc[0].to_dict(),
        stage=stage,
    )

    probabilities = result[
        "status_probabilities"
    ]

    required_classes = {
        "Graduate",
        "Dropout",
        "Enrolled",
    }

    assert set(probabilities.keys()) == required_classes

    values = [
        float(probabilities[name])
        for name in required_classes
    ]

    assert all(
        np.isfinite(value)
        for value in values
    )

    assert all(
        0 <= value <= 1
        for value in values
    )

    assert sum(values) == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert result["predicted_status"] in required_classes


# ============================================
# TEST 10: BINARY WARNING CONSISTENCY
# ============================================

def test_binary_warning(
    engine,
    validator,
    student,
):

    clean = validator.validate(
        student,
        stage="B",
    )

    result = engine.predict_one(
        clean.iloc[0].to_dict(),
        stage="B",
    )

    risk = float(
        result["dropout_risk_score"]
    )

    threshold = float(
        result["threshold"]
    )

    warning = result["warning"]

    assert np.isfinite(risk)

    assert 0 <= risk <= 1

    assert threshold == pytest.approx(
        0.22,
        abs=1e-10,
    )

    assert bool(warning) == (
        risk >= threshold
    )


# ============================================
# EXTRA: DETERMINISTIC PREDICTIONS
# ============================================

def test_repeated_prediction_is_stable(
    engine,
    validator,
    student,
):

    clean = validator.validate(
        student,
        stage="B",
    )

    record = clean.iloc[0].to_dict()

    first = engine.predict_one(
        record,
        stage="B",
    )

    second = engine.predict_one(
        record,
        stage="B",
    )

    assert (
        first["predicted_status"]
        == second["predicted_status"]
    )

    for label in [
        "Graduate",
        "Enrolled",
        "Dropout",
    ]:

        first_probability = first[
            "status_probabilities"
        ][label]

        second_probability = second[
            "status_probabilities"
        ][label]

        assert first_probability == pytest.approx(
            second_probability,
            abs=1e-12,
        )

    assert first[
        "dropout_risk_score"
    ] == pytest.approx(
        second["dropout_risk_score"],
        abs=1e-12,
    )
