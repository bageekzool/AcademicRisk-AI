
from io import BytesIO

import pandas as pd
import pytest

from src.batch_csv_schema import (
    TURKISH_COLUMNS,
    create_turkish_template,
    normalize_uploaded_csv,
    create_turkish_preview,
)
from src.prediction_engine import PredictionEngine
from src.student_prediction_page import CATEGORY_MAPS


# ==========================================
# TEST FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def engine():
    return PredictionEngine()


@pytest.fixture(params=["A", "B"])
def stage_features(request, engine):
    stage = request.param
    features = list(
        engine.get_required_features(stage)
    )
    return stage, features


def make_sample(features):
    """
    Synthetic test data for CSV schema tests.
    These values are not sent to the model.
    """
    return pd.DataFrame([
        {
            feature: index + 1
            for index, feature in enumerate(features)
        }
    ])


# ==========================================
# TEST 1: STAGE FEATURE COUNTS
# ==========================================

def test_stage_feature_count(stage_features):
    stage, features = stage_features

    expected = 8 if stage == "A" else 14

    assert len(features) == expected


# ==========================================
# TEST 2: TURKISH TEMPLATE HEADERS
# ==========================================

def test_turkish_template_columns(stage_features):
    _, features = stage_features

    content = create_turkish_template(features)

    df = pd.read_csv(
        BytesIO(content),
        encoding="utf-8-sig",
    )

    assert list(df.columns) == [
        TURKISH_COLUMNS[feature]
        for feature in features
    ]

    assert df.empty


# ==========================================
# TEST 3: UTF-8 BOM
# ==========================================

def test_template_encoding(stage_features):
    _, features = stage_features

    content = create_turkish_template(features)

    assert content.startswith(b"\xef\xbb\xbf")


# ==========================================
# TEST 4: TURKISH COLUMN NORMALIZATION
# ==========================================

def test_turkish_to_english(stage_features):
    _, features = stage_features

    original = make_sample(features)

    turkish = original.rename(
        columns={
            feature: TURKISH_COLUMNS[feature]
            for feature in features
        }
    )

    restored = normalize_uploaded_csv(
        turkish,
        features,
    )

    assert all(
        feature in restored.columns
        for feature in features
    )

    pd.testing.assert_frame_equal(
        restored[features],
        original[features],
    )


# ==========================================
# TEST 5: LEGACY ENGLISH COLUMNS
# ==========================================

def test_english_columns_supported(stage_features):
    _, features = stage_features

    original = make_sample(features)

    normalized = normalize_uploaded_csv(
        original,
        features,
    )

    pd.testing.assert_frame_equal(
        normalized[features],
        original[features],
    )


# ==========================================
# TEST 6: MIXED TURKISH / ENGLISH
# ==========================================

def test_mixed_language_columns(stage_features):
    _, features = stage_features

    original = make_sample(features)

    rename_map = {
        feature: TURKISH_COLUMNS[feature]
        for feature in features[::2]
    }

    mixed = original.rename(columns=rename_map)

    normalized = normalize_uploaded_csv(
        mixed,
        features,
    )

    pd.testing.assert_frame_equal(
        normalized[features],
        original[features],
    )


# ==========================================
# TEST 7: MISSING COLUMN REJECTED
# ==========================================

def test_missing_column(stage_features):
    _, features = stage_features

    df = make_sample(features).drop(
        columns=[features[0]]
    )

    with pytest.raises(ValueError):
        normalize_uploaded_csv(df, features)


# ==========================================
# TEST 8: DUPLICATE COLUMNS REJECTED
# ==========================================

def test_duplicate_columns(stage_features):
    _, features = stage_features

    df = make_sample(features)

    df[features[0] + " "] = 999

    with pytest.raises(ValueError):
        normalize_uploaded_csv(df, features)


# ==========================================
# TEST 9: TURKISH + ENGLISH CONFLICT
# ==========================================

def test_same_field_twice(stage_features):
    _, features = stage_features

    df = make_sample(features)

    first = features[0]

    df[TURKISH_COLUMNS[first]] = df[first]

    with pytest.raises(ValueError):
        normalize_uploaded_csv(df, features)


# ==========================================
# TEST 10: INPUT DATA NOT MUTATED
# ==========================================

def test_original_dataframe_unchanged(stage_features):
    _, features = stage_features

    original = make_sample(features)

    before = original.copy(deep=True)

    normalize_uploaded_csv(
        original,
        features,
    )

    pd.testing.assert_frame_equal(
        original,
        before,
    )


# ==========================================
# TEST 11: TURKISH PREVIEW
# ==========================================

def test_turkish_preview(stage_features):
    _, features = stage_features

    df = make_sample(features)

    preview = create_turkish_preview(
        df,
        features,
    )

    assert list(preview.columns) == [
        TURKISH_COLUMNS[feature]
        for feature in features
    ]


# ==========================================
# TEST 12: CATEGORY DESCRIPTION TO CODE
# ==========================================

def test_category_description_to_code():
    feature = next(iter(CATEGORY_MAPS))

    mapping = CATEGORY_MAPS[feature]

    code, description = next(
        iter(mapping.items())
    )

    df = pd.DataFrame({
        TURKISH_COLUMNS[feature]: [
            description
        ]
    })

    normalized = normalize_uploaded_csv(
        df,
        [feature],
    )

    assert int(normalized.iloc[0][feature]) == int(code)


# ==========================================
# TEST 13: UNKNOWN CATEGORY PRESERVED
# ==========================================

def test_unknown_category_not_silently_changed():
    feature = next(iter(CATEGORY_MAPS))

    invalid_value = (
        "__BILINMEYEN_KATEGORI__"
    )

    df = pd.DataFrame({
        TURKISH_COLUMNS[feature]: [
            invalid_value
        ]
    })

    normalized = normalize_uploaded_csv(
        df,
        [feature],
    )

    # Schema conversion must not invent
    # a valid category code.
    # The existing input validator handles
    # invalid model values later.

    assert (
        normalized.iloc[0][feature]
        == invalid_value
    )
