
from io import BytesIO

import pandas as pd

from src.student_prediction_page import CATEGORY_MAPS


# ============================================
# TURKISH COLUMN NAMES
# ============================================

TURKISH_COLUMNS = {
    "Application mode": "Başvuru Şekli",
    "Application order": "Başvuru Tercih Sırası",
    "Course": "Bölüm / Program",
    "Daytime/evening attendance": "Öğretim Türü",
    "Previous qualification": "Önceki Eğitim Niteliği",
    "Previous qualification (grade)": "Önceki Eğitim Notu",
    "Admission grade": "Üniversiteye Giriş Notu",
    "Age at enrollment": "Kayıt Yaşı",

    "Curricular units 1st sem (credited)":
        "Sayılan / Muaf Tutulan Ders",

    "Curricular units 1st sem (enrolled)":
        "Kayıtlı Ders Sayısı",

    "Curricular units 1st sem (evaluations)":
        "Değerlendirme Sayısı",

    "Curricular units 1st sem (approved)":
        "Başarılı Ders Sayısı",

    "Curricular units 1st sem (grade)":
        "1. Dönem Not Ortalaması",

    "Curricular units 1st sem (without evaluations)":
        "Değerlendirmesi Olmayan Ders Sayısı",
}

ATTENDANCE = {
    1: "Gündüz Öğretimi",
    0: "Akşam Öğretimi",
}


# ============================================
# TURKISH TEMPLATE
# ============================================

def create_turkish_template(features):

    columns = []

    for feature in features:

        if feature not in TURKISH_COLUMNS:
            raise ValueError(
                f"Türkçe başlık bulunamadı: {feature}"
            )

        columns.append(
            TURKISH_COLUMNS[feature]
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Tekrarlanan Türkçe sütun adı."
        )

    template = pd.DataFrame(
        columns=columns
    )

    return template.to_csv(
        index=False
    ).encode("utf-8-sig")


# ============================================
# CATEGORY GUIDE
# ============================================

def create_category_guide(features, categories):

    rows = []

    for feature in features:

        if feature in CATEGORY_MAPS:

            descriptions = CATEGORY_MAPS[feature]

            allowed_codes = categories.get(
                feature,
                []
            )

            for code in sorted(
                int(value)
                for value in allowed_codes
            ):

                if code not in descriptions:
                    raise ValueError(
                        f"{feature}: açıklaması "
                        f"olmayan kod {code}"
                    )

                rows.append({
                    "Alan": TURKISH_COLUMNS[feature],
                    "Kod": code,
                    "Açıklama": descriptions[code],
                })

        elif feature == "Daytime/evening attendance":

            for code, description in ATTENDANCE.items():

                rows.append({
                    "Alan": TURKISH_COLUMNS[feature],
                    "Kod": code,
                    "Açıklama": description,
                })

    guide = pd.DataFrame(
        rows,
        columns=["Alan", "Kod", "Açıklama"],
    )

    return guide.to_csv(
        index=False
    ).encode("utf-8-sig")


# ============================================
# CONVERT CATEGORY VALUE TO MODEL CODE
# ============================================

def convert_category_value(value, mapping):

    if pd.isna(value):
        return value

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()

    if not text:
        return value

    # Numeric code written as text.
    try:
        numeric = float(text)

        if numeric.is_integer():
            return int(numeric)

    except ValueError:
        pass

    # Accept both:
    # "Bilişim Mühendisliği"
    # "Bilişim Mühendisliği (9119)"

    matches = []

    for code, description in mapping.items():

        accepted = {
            description,
            f"{description} ({code})",
        }

        if text in accepted:
            matches.append(code)

    if len(matches) == 1:
        return matches[0]

    # Unknown values remain unchanged.
    # The existing validator must reject them.
    return value


# ============================================
# NORMALIZE UPLOADED CSV
# ============================================

def normalize_uploaded_csv(df, features):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "CSV verisi DataFrame olmalıdır."
        )

    normalized = df.copy()

    # Remove whitespace from CSV header names.
    normalized.columns = [
        str(column).strip()
        for column in normalized.columns
    ]

    if normalized.columns.duplicated().any():
        raise ValueError(
            "CSV dosyasında tekrarlanan "
            "sütun adları var."
        )

    rename_map = {}

    for feature in features:

        turkish = TURKISH_COLUMNS[feature]

        has_english = (
            feature in normalized.columns
        )

        has_turkish = (
            turkish in normalized.columns
        )

        if has_english and has_turkish:
            raise ValueError(
                f"Aynı alan iki kez bulunuyor: "
                f"{turkish}"
            )

        if has_turkish:
            rename_map[turkish] = feature

    normalized = normalized.rename(
        columns=rename_map
    )

    missing = [
        TURKISH_COLUMNS[feature]
        for feature in features
        if feature not in normalized.columns
    ]

    if missing:
        raise ValueError(
            "Eksik CSV sütunları: "
            + ", ".join(missing)
        )

    # Convert named categories to model codes.
    for feature in features:

        if feature in CATEGORY_MAPS:

            mapping = CATEGORY_MAPS[feature]

            normalized[feature] = (
                normalized[feature].map(
                    lambda value: convert_category_value(
                        value,
                        mapping,
                    )
                )
            )

        elif feature == "Daytime/evening attendance":

            normalized[feature] = (
                normalized[feature].map(
                    lambda value: convert_category_value(
                        value,
                        ATTENDANCE,
                    )
                )
            )

    return normalized


# ============================================
# TURKISH PREVIEW
# ============================================

def create_turkish_preview(df, features):

    return df[features].rename(
        columns={
            feature: TURKISH_COLUMNS[feature]
            for feature in features
        }
    )


# ============================================
# SIMPLE SELF TEST
# ============================================

def main():

    features = [
        "Application mode",
        "Course",
        "Daytime/evening attendance",
        "Age at enrollment",
    ]

    csv_data = create_turkish_template(features)

    template = pd.read_csv(
        BytesIO(csv_data)
    )

    assert list(template.columns) == [
        TURKISH_COLUMNS[feature]
        for feature in features
    ]

    example = pd.DataFrame([{
        "Başvuru Şekli":
            "1. Aşama - Genel Kontenjan (1)",
        "Bölüm / Program":
            "Bilişim Mühendisliği (9119)",
        "Öğretim Türü":
            "Gündüz Öğretimi",
        "Kayıt Yaşı": 21,
    }])

    normalized = normalize_uploaded_csv(
        example,
        features,
    )

    assert normalized.iloc[0][
        "Application mode"
    ] == 1

    assert normalized.iloc[0][
        "Course"
    ] == 9119

    assert normalized.iloc[0][
        "Daytime/evening attendance"
    ] == 1

    assert normalized.iloc[0][
        "Age at enrollment"
    ] == 21

    print("[SUCCESS] Turkish CSV template.")
    print("[SUCCESS] Category name conversion.")
    print("[SUCCESS] Original model codes preserved.")


if __name__ == "__main__":
    main()
