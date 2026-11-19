
import ast
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.batch_prediction_page as batch_page

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator
from src.shap_engine import ShapEngine
from src.student_prediction_page import (
    render_student_prediction,
)
from src.batch_prediction_page import (
    render_batch_prediction,
)


# ============================================
# PROJECT CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parent.parent

APP_PATH = ROOT / "app.py"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"

MODEL_FILES = [
    ROOT / "models" / "random_forest_A.joblib",
    ROOT / "models" /
    "multiclass_logistic_B_balanced.joblib",
    ROOT / "models" /
    "binary_logistic_B_threshold.joblib",
    ROOT / "models" /
    "binary_B_shap_reference.joblib",
]

PAGES = [
    "Ana Sayfa",
    "Öğrenci Tahmini",
    "Toplu Tahmin",
    "Model Performansı",
    "Erken Uyarı",
    "Sistem Hakkında",
]

CLASS_LABELS = {
    "Graduate": "Mezuniyet",
    "Enrolled": "Öğrenime Devam",
    "Dropout": "Okuldan Ayrılma",
}


# ============================================
# HELPERS
# ============================================

def file_hash(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


class FakeProgress:

    def progress(self, value, text=None):
        return self

    def empty(self):
        pass


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
def shap_engine(engine):
    return ShapEngine(engine)


@pytest.fixture(scope="module")
def samples():

    data = pd.read_csv(TRAIN_PATH)

    assert len(data) >= 5

    # Only functional checks on TRAIN records.
    return data.head(5).copy()


@pytest.fixture
def fake_streamlit_progress(monkeypatch):

    monkeypatch.setattr(
        batch_page.st,
        "progress",
        lambda *args, **kwargs: FakeProgress(),
    )


@pytest.fixture(scope="module", autouse=True)
def protect_model_artifacts():

    original_hashes = {}

    for path in MODEL_FILES:

        assert path.exists(), (
            f"Missing artifact: {path}"
        )

        original_hashes[path] = file_hash(path)

    yield

    # Tests must not modify frozen models.
    for path, original_hash in (
        original_hashes.items()
    ):

        assert path.exists()

        assert file_hash(path) == original_hash, (
            f"Model artifact changed: {path}"
        )


def student_record(
    samples,
    engine,
    stage,
    index=0,
):

    features = engine.get_required_features(stage)

    return samples.iloc[index][
        features
    ].to_dict()


# ============================================
# TEST 01: APPLICATION SYNTAX
# ============================================

def test_app_syntax():

    source = APP_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source,
        filename=str(APP_PATH),
    )

    assert isinstance(
        tree,
        ast.Module,
    )


# ============================================
# TEST 02: NAVIGATION LABELS
# ============================================

def test_navigation_pages_exist():

    source = APP_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }

    for page in PAGES:

        assert page in strings, (
            f"Missing navigation label: {page}"
        )


# ============================================
# TEST 03: PAGE FUNCTIONS
# ============================================

def test_page_functions_importable():

    assert callable(
        render_student_prediction
    )

    assert callable(
        render_batch_prediction
    )


# ============================================
# TEST 04: PAGE CALLS IN APP
# ============================================

def test_page_functions_used_by_app():

    source = APP_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    called_functions = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }

    assert (
        "render_student_prediction"
        in called_functions
    )

    assert (
        "render_batch_prediction"
        in called_functions
    )


# ============================================
# TEST 05: STAGE SCHEMAS
# ============================================

@pytest.mark.parametrize(
    "stage,expected_count",
    [
        ("A", 8),
        ("B", 14),
    ],
)
def test_stage_schema_integration(
    engine,
    validator,
    stage,
    expected_count,
):

    features = engine.get_required_features(stage)

    assert len(features) == expected_count

    categories = (
        validator.get_allowed_categories(stage)
    )

    assert set(categories).issubset(
        set(features)
    )

    assert len(categories["Course"]) == 17

    assert len(
        categories["Application mode"]
    ) == 18

    assert len(
        categories["Previous qualification"]
    ) == 17


# ============================================
# TEST 06: CSV ROUND TRIP
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_csv_round_trip(
    engine,
    samples,
    stage,
):

    features = engine.get_required_features(stage)

    original = samples[features].copy()

    csv_bytes = original.to_csv(
        index=False
    ).encode("utf-8-sig")

    restored = pd.read_csv(
        BytesIO(csv_bytes)
    )

    assert list(restored.columns) == list(features)

    pd.testing.assert_frame_equal(
        original.reset_index(drop=True),
        restored.reset_index(drop=True),
        check_dtype=False,
    )


# ============================================
# TEST 07: END-TO-END CSV PREDICTION
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_csv_prediction_integration(
    engine,
    validator,
    samples,
    fake_streamlit_progress,
    stage,
):

    features = engine.get_required_features(stage)

    csv_bytes = samples[features].to_csv(
        index=False
    ).encode("utf-8-sig")

    uploaded = pd.read_csv(
        BytesIO(csv_bytes)
    )

    results, errors = batch_page.process_batch(
        uploaded,
        stage,
        engine,
        validator,
    )

    assert errors.empty
    assert len(results) == 5

    for index in range(5):

        student = uploaded.iloc[
            index
        ].to_dict()

        validated = validator.validate(
            student,
            stage=stage,
        )

        single = engine.predict_one(
            validated.iloc[0].to_dict(),
            stage=stage,
        )

        batch = results.iloc[index]

        assert batch[
            "Tahmin Edilen Durum"
        ] == CLASS_LABELS[
            single["predicted_status"]
        ]

        assert float(
            batch["Ayrılma Olasılığı"]
        ) == pytest.approx(
            single["status_probabilities"][
                "Dropout"
            ],
            abs=1e-10,
        )

        if stage == "B":

            assert float(
                batch["Erken Uyarı Risk Skoru"]
            ) == pytest.approx(
                single["dropout_risk_score"],
                abs=1e-10,
            )


# ============================================
# TEST 08: INVALID ROW INTEGRATION
# ============================================

def test_invalid_row_isolation_integration(
    engine,
    validator,
    samples,
    fake_streamlit_progress,
):

    features = engine.get_required_features("B")

    data = samples[features].copy()

    data.loc[
        data.index[1],
        "Course",
    ] = 999999

    results, errors = batch_page.process_batch(
        data,
        "B",
        engine,
        validator,
    )

    assert len(results) == 4
    assert len(errors) == 1

    assert errors.iloc[0][
        "CSV Satır No"
    ] == 3

    assert (
        3 not in results["CSV Satır No"].tolist()
    )


# ============================================
# TEST 09: PREDICTION + SHAP INTEGRATION
# ============================================

@pytest.mark.parametrize(
    "index",
    [0, 1, 2, 3, 4],
)
def test_prediction_shap_integration(
    engine,
    validator,
    shap_engine,
    samples,
    index,
):

    student = student_record(
        samples,
        engine,
        "B",
        index,
    )

    clean = validator.validate(
        student,
        stage="B",
    )

    record = clean.iloc[0].to_dict()

    prediction = engine.predict_one(
        record,
        stage="B",
    )

    explanation = shap_engine.explain_one(
        record
    )

    assert explanation[
        "dropout_probability"
    ] == pytest.approx(
        prediction["dropout_risk_score"],
        abs=1e-8,
    )

    assert explanation["warning"] == bool(
        prediction["warning"]
    )

    assert len(
        explanation["contributions"]
    ) == 14

    assert np.isfinite(
        explanation["contributions"][
            "shap_log_odds"
        ]
    ).all()


# ============================================
# TEST 10: INVALID INPUT DOES NOT PREDICT
# ============================================

def test_invalid_input_rejected_before_prediction(
    engine,
    validator,
    samples,
    monkeypatch,
):

    student = student_record(
        samples,
        engine,
        "B",
    )

    student["Course"] = 999999

    call_count = {"count": 0}

    def forbidden_prediction(*args, **kwargs):

        call_count["count"] += 1

        raise AssertionError(
            "Model should not be called."
        )

    monkeypatch.setattr(
        engine,
        "predict_one",
        forbidden_prediction,
    )

    with pytest.raises(ValueError):

        clean = validator.validate(
            student,
            stage="B",
        )

        engine.predict_one(
            clean.iloc[0].to_dict(),
            stage="B",
        )

    assert call_count["count"] == 0


# ============================================
# TEST 11: FROZEN WARNING THRESHOLD
# ============================================

def test_frozen_warning_threshold(
    engine,
    shap_engine,
):

    artifact = engine.models[
        "binary_warning_B"
    ]["artifact"]

    assert float(
        artifact["threshold"]
    ) == pytest.approx(0.22)

    assert shap_engine.threshold == pytest.approx(
        0.22
    )

    assert float(
        shap_engine.reference["threshold"]
    ) == pytest.approx(0.22)


# ============================================
# TEST 12: MODEL FILE INTEGRITY
# ============================================

def test_model_files_present():

    for path in MODEL_FILES:

        assert path.is_file()

        assert path.stat().st_size > 0
