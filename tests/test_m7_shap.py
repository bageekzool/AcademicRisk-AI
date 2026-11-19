
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scipy.special import expit

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator
from src.shap_engine import ShapEngine


# ============================================
# PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"

SAMPLE_INDICES = [0, 1, 2, 3, 4]


# ============================================
# SHARED FIXTURES
# ============================================

@pytest.fixture(scope="module")
def engine():
    return PredictionEngine()


@pytest.fixture(scope="module")
def validator(engine):
    return StudentInputValidator(engine)


@pytest.fixture(scope="module")
def shap_engine(engine):
    return ShapEngine(engine)


@pytest.fixture(scope="module")
def samples(engine):

    data = pd.read_csv(TRAIN_PATH)

    assert len(data) >= 5

    features = engine.get_required_features("B")

    return data[features].head(5).copy()


def prepare_student(samples, validator, index):

    student = samples.iloc[index].to_dict()

    clean = validator.validate(
        student,
        stage="B",
    )

    return clean.iloc[0].to_dict()


# ============================================
# TEST 01: SHAP REFERENCE
# ============================================

def test_shap_reference(shap_engine):

    reference = shap_engine.reference

    assert reference["model"] == "binary_warning_B"

    assert reference["background_size"] == 150

    assert len(shap_engine.features) == 14

    assert len(shap_engine.transformed_names) == 63

    assert len(shap_engine.reference_mean) == 63

    assert np.isfinite(
        shap_engine.reference_mean
    ).all()

    assert np.isfinite(
        shap_engine.base_log_odds
    )


# ============================================
# TEST 02: PROBABILITY RECONSTRUCTION
# ============================================

@pytest.mark.parametrize("index", SAMPLE_INDICES)
def test_shap_reconstructs_probability(
    engine,
    validator,
    shap_engine,
    samples,
    index,
):

    student = prepare_student(
        samples,
        validator,
        index,
    )

    explanation = shap_engine.explain_one(
        student
    )

    prediction = engine.predict_one(
        student,
        stage="B",
    )

    shap_risk = float(
        explanation["dropout_probability"]
    )

    engine_risk = float(
        prediction["dropout_risk_score"]
    )

    assert shap_risk == pytest.approx(
        engine_risk,
        abs=1e-8,
    )

    assert explanation[
        "reconstruction_error"
    ] < 1e-8


# ============================================
# TEST 03: ORIGINAL FEATURE AGGREGATION
# ============================================

@pytest.mark.parametrize("index", SAMPLE_INDICES)
def test_shap_feature_aggregation(
    validator,
    shap_engine,
    samples,
    index,
):

    student = prepare_student(
        samples,
        validator,
        index,
    )

    explanation = shap_engine.explain_one(
        student
    )

    contributions = explanation[
        "contributions"
    ]

    assert len(contributions) == 14

    assert (
        contributions["feature"].is_unique
    )

    assert set(
        contributions["feature"]
    ) == set(shap_engine.features)

    assert np.isfinite(
        contributions["shap_log_odds"]
    ).all()

    total_contribution = float(
        contributions["shap_log_odds"].sum()
    )

    reconstructed = expit(
        explanation["base_log_odds"]
        + total_contribution
    )

    assert reconstructed == pytest.approx(
        explanation["dropout_probability"],
        abs=1e-8,
    )


# ============================================
# TEST 04: SHAP DIRECTION
# ============================================

@pytest.mark.parametrize("index", SAMPLE_INDICES)
def test_shap_direction(
    validator,
    shap_engine,
    samples,
    index,
):

    student = prepare_student(
        samples,
        validator,
        index,
    )

    explanation = shap_engine.explain_one(
        student
    )

    contributions = explanation[
        "contributions"
    ]

    for _, row in contributions.iterrows():

        value = float(
            row["shap_log_odds"]
        )

        expected_direction = (
            "Artırıyor"
            if value > 0
            else (
                "Azaltıyor"
                if value < 0
                else "Nötr"
            )
        )

        assert row["direction"] == expected_direction

        assert float(
            row["abs_shap"]
        ) == pytest.approx(
            abs(value),
            abs=1e-10,
        )


# ============================================
# TEST 05: BASE VALUE CONSISTENCY
# ============================================

def test_shap_base_value(shap_engine):

    expected = (
        shap_engine.intercept
        + np.dot(
            shap_engine.coefficients,
            shap_engine.reference_mean,
        )
    )

    assert shap_engine.base_log_odds == pytest.approx(
        expected,
        abs=1e-8,
    )

    assert 0 < expit(expected) < 1


# ============================================
# TEST 06: WARNING DECISION
# ============================================

@pytest.mark.parametrize("index", SAMPLE_INDICES)
def test_shap_warning_matches_model(
    engine,
    validator,
    shap_engine,
    samples,
    index,
):

    student = prepare_student(
        samples,
        validator,
        index,
    )

    prediction = engine.predict_one(
        student,
        stage="B",
    )

    explanation = shap_engine.explain_one(
        student
    )

    assert explanation["threshold"] == pytest.approx(
        0.22,
        abs=1e-10,
    )

    assert bool(
        explanation["warning"]
    ) == bool(prediction["warning"])

    assert bool(
        explanation["warning"]
    ) == (
        explanation["dropout_probability"]
        >= explanation["threshold"]
    )


# ============================================
# TEST 07: REPEATABILITY
# ============================================

def test_shap_repeatability(
    validator,
    shap_engine,
    samples,
):

    student = prepare_student(
        samples,
        validator,
        0,
    )

    first = shap_engine.explain_one(student)
    second = shap_engine.explain_one(student)

    assert first[
        "dropout_probability"
    ] == pytest.approx(
        second["dropout_probability"],
        abs=1e-12,
    )

    pd.testing.assert_frame_equal(
        first["contributions"],
        second["contributions"],
    )


# ============================================
# TEST 08: INVALID INPUT REJECTION
# ============================================

def test_shap_rejects_invalid_student(
    validator,
    shap_engine,
    samples,
):

    student = prepare_student(
        samples,
        validator,
        0,
    )

    student[
        "Curricular units 1st sem (grade)"
    ] = 99.0

    with pytest.raises(ValueError):

        shap_engine.explain_one(
            student
        )


# ============================================
# TEST 09: ORIGINAL VALUES
# ============================================

def test_shap_preserves_original_values(
    validator,
    shap_engine,
    samples,
):

    student = prepare_student(
        samples,
        validator,
        0,
    )

    explanation = shap_engine.explain_one(
        student
    )

    contributions = explanation[
        "contributions"
    ]

    for _, row in contributions.iterrows():

        feature = row["feature"]

        assert float(
            row["original_value"]
        ) == pytest.approx(
            float(student[feature]),
            abs=1e-10,
        )
