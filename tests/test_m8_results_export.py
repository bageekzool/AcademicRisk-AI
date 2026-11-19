
from io import BytesIO

import pandas as pd
import pytest

from src.batch_prediction_page import (
    create_results_csv,
    prepare_results_export,
)


# ==========================================
# TEST DATA - STAGE A
# ==========================================

@pytest.fixture
def results_a():
    return pd.DataFrame([
        {
            "CSV Satır No": 2,
            "Tahmin Edilen Durum": "Mezuniyet",
            "Mezuniyet Olasılığı": 0.7325,
            "Devam Olasılığı": 0.1675,
            "Ayrılma Olasılığı": 0.10,
        },
    ])


# ==========================================
# TEST DATA - STAGE B
# ==========================================

@pytest.fixture
def results_b():
    return pd.DataFrame([
        {
            "CSV Satır No": 2,
            "Tahmin Edilen Durum": "Okuldan Ayrılma",
            "Mezuniyet Olasılığı": 0.10,
            "Devam Olasılığı": 0.20,
            "Ayrılma Olasılığı": 0.70,
            "Erken Uyarı Risk Skoru": 0.81,
            "Erken Uyarı": "Evet",
        },
    ])


# ==========================================
# TEST 1: STAGE A PERCENTAGES
# ==========================================

def test_stage_a_percentages(results_a):

    exported = prepare_results_export(
        results_a,
        "A",
    )

    assert exported.loc[
        0, "Mezuniyet Olasılığı (%)"
    ] == pytest.approx(73.25)

    assert exported.loc[
        0, "Devam Olasılığı (%)"
    ] == pytest.approx(16.75)

    assert exported.loc[
        0, "Ayrılma Olasılığı (%)"
    ] == pytest.approx(10.0)


# ==========================================
# TEST 2: STAGE B RISK SCORE
# ==========================================

def test_stage_b_risk_score(results_b):

    exported = prepare_results_export(
        results_b,
        "B",
    )

    assert exported.loc[
        0, "Erken Uyarı Risk Skoru (%)"
    ] == pytest.approx(81.0)

    assert exported.loc[
        0, "Erken Uyarı"
    ] == "Evet"


# ==========================================
# TEST 3: STAGE LABEL
# ==========================================

@pytest.mark.parametrize(
    "stage, fixture_name",
    [
        ("A", "results_a"),
        ("B", "results_b"),
    ],
)
def test_stage_label(
    stage,
    fixture_name,
    request,
):

    results = request.getfixturevalue(
        fixture_name
    )

    exported = prepare_results_export(
        results,
        stage,
    )

    assert exported.loc[
        0, "Tahmin Aşaması"
    ] == f"Aşama {stage}"


# ==========================================
# TEST 4: SOURCE DATA MUST NOT CHANGE
# ==========================================

def test_original_results_unchanged(results_b):

    original = results_b.copy(deep=True)

    prepare_results_export(
        results_b,
        "B",
    )

    pd.testing.assert_frame_equal(
        results_b,
        original,
    )


# ==========================================
# TEST 5: CSV ENCODING AND ROUND TRIP
# ==========================================

@pytest.mark.parametrize(
    "stage, fixture_name",
    [
        ("A", "results_a"),
        ("B", "results_b"),
    ],
)
def test_csv_round_trip(
    stage,
    fixture_name,
    request,
):

    results = request.getfixturevalue(
        fixture_name
    )

    content = create_results_csv(
        results,
        stage,
    )

    assert content.startswith(
        b"\xef\xbb\xbf"
    )

    restored = pd.read_csv(
        BytesIO(content),
        encoding="utf-8-sig",
    )

    assert len(restored) == len(results)

    assert restored.loc[
        0, "Tahmin Aşaması"
    ] == f"Aşama {stage}"


# ==========================================
# TEST 6: INVALID STAGE
# ==========================================

def test_invalid_stage(results_a):

    with pytest.raises(ValueError):
        prepare_results_export(
            results_a,
            "C",
        )


# ==========================================
# TEST 7: MISSING COLUMN
# ==========================================

def test_missing_column(results_a):

    invalid = results_a.drop(
        columns=["Mezuniyet Olasılığı"]
    )

    with pytest.raises(ValueError):
        prepare_results_export(
            invalid,
            "A",
        )


# ==========================================
# TEST 8: INVALID PROBABILITY
# ==========================================

@pytest.mark.parametrize(
    "invalid_value",
    [-0.1, 1.1, None],
)
def test_invalid_probability(
    results_a,
    invalid_value,
):

    results_a.loc[
        0, "Mezuniyet Olasılığı"
    ] = invalid_value

    with pytest.raises(ValueError):
        prepare_results_export(
            results_a,
            "A",
        )


# ==========================================
# TEST 9: STAGE A HAS NO BINARY RISK
# ==========================================

def test_stage_a_has_no_binary_risk(results_a):

    exported = prepare_results_export(
        results_a,
        "A",
    )

    assert (
        "Erken Uyarı Risk Skoru (%)"
        not in exported.columns
    )

    assert (
        "Erken Uyarı"
        not in exported.columns
    )


# ==========================================
# TEST 10: OUTPUT COLUMN ORDER
# ==========================================

def test_stage_b_column_order(results_b):

    exported = prepare_results_export(
        results_b,
        "B",
    )

    assert list(exported.columns) == [
        "CSV Satır No",
        "Tahmin Aşaması",
        "Tahmin Edilen Durum",
        "Mezuniyet Olasılığı (%)",
        "Devam Olasılığı (%)",
        "Ayrılma Olasılığı (%)",
        "Erken Uyarı Risk Skoru (%)",
        "Erken Uyarı",
    ]
