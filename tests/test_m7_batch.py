
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.batch_prediction_page as batch_page

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator


# ============================================
# PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    ROOT / "data" / "splits" / "train.csv"
)

CLASS_LABELS = {
    "Graduate": "Mezuniyet",
    "Enrolled": "Öğrenime Devam",
    "Dropout": "Okuldan Ayrılma",
}


# ============================================
# STREAMLIT PROGRESS MOCK
# ============================================

class FakeProgress:

    def __init__(self):
        self.values = []
        self.cleared = False

    def progress(self, value, text=None):
        self.values.append(float(value))
        return self

    def empty(self):
        self.cleared = True


@pytest.fixture
def fake_progress(monkeypatch):

    progress = FakeProgress()

    monkeypatch.setattr(
        batch_page.st,
        "progress",
        lambda *args, **kwargs: progress,
    )

    return progress


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
def training_data():

    if not TRAIN_PATH.exists():
        pytest.fail(
            f"Training data missing: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    assert len(df) >= 5

    return df.head(5).copy()


def get_batch_data(training_data, engine, stage):

    features = list(
        engine.get_required_features(stage)
    )

    return training_data[features].copy()


# ============================================
# TEST 01: CSV TEMPLATE
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_csv_template(engine, stage):

    features = list(
        engine.get_required_features(stage)
    )

    csv_bytes = batch_page.create_template(
        features
    )

    assert isinstance(csv_bytes, bytes)

    import io

    template = pd.read_csv(
        io.BytesIO(csv_bytes)
    )

    assert list(template.columns) == features

    assert len(template) == 0


# ============================================
# TEST 02: VALID BATCH
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_valid_batch(
    engine,
    validator,
    training_data,
    fake_progress,
    stage,
):

    data = get_batch_data(
        training_data,
        engine,
        stage,
    )

    results, errors = batch_page.process_batch(
        df=data,
        stage=stage,
        engine=engine,
        validator=validator,
    )

    assert len(results) == 5
    assert errors.empty

    assert results[
        "CSV Satır No"
    ].tolist() == [2, 3, 4, 5, 6]

    assert results[
        "Tahmin Edilen Durum"
    ].notna().all()

    probability_columns = [
        "Mezuniyet Olasılığı",
        "Devam Olasılığı",
        "Ayrılma Olasılığı",
    ]

    probabilities = results[
        probability_columns
    ].to_numpy(dtype=float)

    assert np.isfinite(probabilities).all()

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )

    if stage == "B":

        assert "Erken Uyarı" in results.columns

        assert (
            "Erken Uyarı Risk Skoru"
            in results.columns
        )

        assert results[
            "Erken Uyarı"
        ].isin(["Evet", "Hayır"]).all()

    else:

        assert "Erken Uyarı" not in results.columns

        assert (
            "Erken Uyarı Risk Skoru"
            not in results.columns
        )

    assert fake_progress.cleared

    assert fake_progress.values[-1] == pytest.approx(
        1.0
    )


# ============================================
# TEST 03: SINGLE VS BATCH PARITY
# ============================================

@pytest.mark.parametrize("stage", ["A", "B"])
def test_single_batch_parity(
    engine,
    validator,
    training_data,
    fake_progress,
    stage,
):

    data = get_batch_data(
        training_data,
        engine,
        stage,
    )

    batch_results, errors = (
        batch_page.process_batch(
            df=data,
            stage=stage,
            engine=engine,
            validator=validator,
        )
    )

    assert errors.empty
    assert len(batch_results) == len(data)

    for index in range(len(data)):

        student = data.iloc[index].to_dict()

        clean = validator.validate(
            student,
            stage=stage,
        )

        single = engine.predict_one(
            clean.iloc[0].to_dict(),
            stage=stage,
        )

        batch = batch_results.iloc[index]

        expected_status = CLASS_LABELS[
            single["predicted_status"]
        ]

        assert (
            batch["Tahmin Edilen Durum"]
            == expected_status
        )

        mapping = {
            "Graduate": "Mezuniyet Olasılığı",
            "Enrolled": "Devam Olasılığı",
            "Dropout": "Ayrılma Olasılığı",
        }

        for model_label, csv_column in mapping.items():

            assert float(
                batch[csv_column]
            ) == pytest.approx(
                float(
                    single[
                        "status_probabilities"
                    ][model_label]
                ),
                abs=1e-10,
            )

        if stage == "B":

            assert float(
                batch["Erken Uyarı Risk Skoru"]
            ) == pytest.approx(
                float(
                    single["dropout_risk_score"]
                ),
                abs=1e-10,
            )

            expected_warning = (
                "Evet"
                if single["warning"]
                else "Hayır"
            )

            assert (
                batch["Erken Uyarı"]
                == expected_warning
            )


# ============================================
# TEST 04: INVALID ROW ISOLATION
# ============================================

def test_invalid_rows_do_not_stop_batch(
    engine,
    validator,
    training_data,
    fake_progress,
):

    data = get_batch_data(
        training_data,
        engine,
        "B",
    )

    # Row 2 in DataFrame -> CSV row 3
    data.loc[
        data.index[1],
        "Course",
    ] = 999999

    # Row 4 in DataFrame -> CSV row 5
    data.loc[
        data.index[3],
        "Curricular units 1st sem (grade)",
    ] = 85.0

    results, errors = batch_page.process_batch(
        df=data,
        stage="B",
        engine=engine,
        validator=validator,
    )

    assert len(results) == 3
    assert len(errors) == 2

    assert errors[
        "CSV Satır No"
    ].tolist() == [3, 5]

    assert results[
        "CSV Satır No"
    ].tolist() == [2, 4, 6]

    assert errors["Hata"].notna().all()

    assert fake_progress.cleared


# ============================================
# TEST 05: ALL INVALID ROWS
# ============================================

def test_all_invalid_rows(
    engine,
    validator,
    training_data,
    fake_progress,
):

    data = get_batch_data(
        training_data,
        engine,
        "B",
    )

    data["Course"] = 999999

    results, errors = batch_page.process_batch(
        df=data,
        stage="B",
        engine=engine,
        validator=validator,
    )

    assert results.empty
    assert len(errors) == 5

    assert errors[
        "CSV Satır No"
    ].tolist() == [2, 3, 4, 5, 6]

    assert fake_progress.cleared


# ============================================
# TEST 06: WARNING THRESHOLD
# ============================================

def test_batch_warning_threshold(
    engine,
    validator,
    training_data,
    fake_progress,
):

    data = get_batch_data(
        training_data,
        engine,
        "B",
    )

    results, errors = batch_page.process_batch(
        df=data,
        stage="B",
        engine=engine,
        validator=validator,
    )

    assert errors.empty

    threshold = 0.22

    for _, row in results.iterrows():

        risk = float(
            row["Erken Uyarı Risk Skoru"]
        )

        expected = (
            "Evet" if risk >= threshold
            else "Hayır"
        )

        assert row["Erken Uyarı"] == expected
