"""AcademicRisk AI — Turkish batch prediction page."""

from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator
from src.batch_csv_schema import (
    TURKISH_COLUMNS,
    create_turkish_template,
    create_category_guide,
    normalize_uploaded_csv,
    create_turkish_preview,
)

MAX_ROWS = 500
MAX_FILE_MB = 5

STATUS_LABELS = {
    "Graduate": "Mezuniyet",
    "Enrolled": "Öğrenime Devam",
    "Dropout": "Okuldan Ayrılma",
}
PROBABILITY_COLUMNS = [
    "Mezuniyet Olasılığı",
    "Devam Olasılığı",
    "Ayrılma Olasılığı",
]
RISK_COLUMN = "Erken Uyarı Risk Skoru"


@st.cache_resource
def load_batch_services():
    engine = PredictionEngine()
    validator = StudentInputValidator(engine)
    return engine, validator


def create_template(features):
    """Legacy English template (kept for M7 tests)."""
    return pd.DataFrame(columns=features).to_csv(index=False).encode("utf-8-sig")


def prepare_results_export(results, stage):
    """Create a percentage-based Turkish export without mutating model results."""
    if stage not in {"A", "B"}:
        raise ValueError("Tahmin aşaması A veya B olmalıdır.")
    if not isinstance(results, pd.DataFrame):
        raise TypeError("Sonuçlar DataFrame olmalıdır.")

    export = results.copy(deep=True)
    required = ["CSV Satır No", "Tahmin Edilen Durum", *PROBABILITY_COLUMNS]
    if stage == "B":
        required.extend([RISK_COLUMN, "Erken Uyarı"])
    missing = [name for name in required if name not in export.columns]
    if missing:
        raise ValueError("Sonuç dosyasında eksik alanlar: " + ", ".join(missing))

    percentage_columns = [*PROBABILITY_COLUMNS]
    if stage == "B":
        percentage_columns.append(RISK_COLUMN)
    for column in percentage_columns:
        values = pd.to_numeric(export[column], errors="raise")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"Geçersiz olasılık değeri: {column}")
        export[column] = (values * 100).round(2)

    export = export.rename(
        columns={name: f"{name} (%)" for name in percentage_columns}
    )
    export.insert(1, "Tahmin Aşaması", f"Aşama {stage}")
    ordered = [
        "CSV Satır No",
        "Tahmin Aşaması",
        "Tahmin Edilen Durum",
        "Mezuniyet Olasılığı (%)",
        "Devam Olasılığı (%)",
        "Ayrılma Olasılığı (%)",
    ]
    if stage == "B":
        ordered.extend(["Erken Uyarı Risk Skoru (%)", "Erken Uyarı"])
    return export[ordered]


def create_results_csv(results, stage):
    return prepare_results_export(results, stage).to_csv(
        index=False
    ).encode("utf-8-sig")


def process_batch(df, stage, engine, validator):
    """Keep internal probabilities in [0, 1] for backward compatibility."""
    required = list(engine.get_required_features(stage))
    results, errors = [], []
    total = len(df)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame(columns=["CSV Satır No", "Hata"])

    progress = st.progress(0, text="Öğrenciler analiz ediliyor...")
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        csv_row_number = index + 1
        student = {feature: row[feature] for feature in required}
        try:
            clean = validator.validate(student, stage=stage)
            prediction = engine.predict_one(clean.iloc[0].to_dict(), stage=stage)
            probabilities = prediction["status_probabilities"]
            predicted = prediction["predicted_status"]
            output = {
                "CSV Satır No": csv_row_number,
                "Tahmin Edilen Durum": STATUS_LABELS.get(predicted, predicted),
                "Mezuniyet Olasılığı": float(probabilities["Graduate"]),
                "Devam Olasılığı": float(probabilities["Enrolled"]),
                "Ayrılma Olasılığı": float(probabilities["Dropout"]),
            }
            if stage == "B":
                output[RISK_COLUMN] = float(prediction["dropout_risk_score"])
                output["Erken Uyarı"] = "Evet" if prediction["warning"] else "Hayır"
            results.append(output)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"CSV Satır No": csv_row_number, "Hata": str(exc)})
        progress.progress(index / total, text=f"{index}/{total} kayıt işlendi")

    progress.empty()
    return (
        pd.DataFrame(results),
        pd.DataFrame(errors, columns=["CSV Satır No", "Hata"]),
    )


def show_summary(results, errors, stage):
    """Show only metrics, distribution, predictions, and CSV download.

    No explanation paragraphs, captions, or expanders are rendered here.
    """
    st.markdown("### Analiz Özeti")
    successful, failed = len(results), len(errors)
    if stage == "B":
        warning_count = (
            int((results["Erken Uyarı"] == "Evet").sum())
            if not results.empty else 0
        )
        a, b, c = st.columns(3)
        a.metric("Başarılı Kayıt", successful)
        b.metric("Hatalı Kayıt", failed)
        c.metric("Erken Uyarı Sayısı", warning_count)
    else:
        a, b = st.columns(2)
        a.metric("Başarılı Kayıt", successful)
        b.metric("Hatalı Kayıt", failed)

    if results.empty:
        st.info("Geçerli öğrenci kaydı bulunamadı. Hata raporunu inceleyebilirsiniz.")
        return

    st.markdown("### Akademik Durum Dağılımı")
    counts = results["Tahmin Edilen Durum"].value_counts()
    st.bar_chart(counts, horizontal=True, color="#6658DB")

    st.markdown("### Öğrenci Tahmin Sonuçları")
    display = prepare_results_export(results, stage)
    percentage_columns = [name for name in display.columns if name.endswith("(%)")]
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(name, format="%.2f")
            for name in percentage_columns
        },
    )
    st.download_button(
        "Tahmin Sonuçlarını CSV Olarak İndir",
        data=create_results_csv(results, stage),
        file_name=f"academicrisk_tahmin_{stage}.csv",
        mime="text/csv",
        width="stretch",
        key=f"download_predictions_{stage}",
    )


def show_csv_resources(features, validator, stage):
    st.markdown("### 1. CSV Şablonu ve Kod Rehberi")
    st.write(f"Aşama {stage} için {len(features)} özellik gereklidir.")
    left, right = st.columns(2, gap="medium")
    with left:
        st.download_button(
            "Türkçe CSV Şablonunu İndir",
            data=create_turkish_template(features),
            file_name=f"academicrisk_sablon_{stage}.csv",
            mime="text/csv",
            width="stretch",
            key=f"template_{stage}",
        )
    with right:
        st.download_button(
            "Kategori Kodları Rehberini İndir",
            data=create_category_guide(
                features, validator.get_allowed_categories(stage)
            ),
            file_name=f"academicrisk_kod_rehberi_{stage}.csv",
            mime="text/csv",
            width="stretch",
            key=f"category_guide_{stage}",
        )

    with st.expander("Gerekli CSV Sütunlarını Gör"):
        st.dataframe(
            pd.DataFrame({
                "Türkçe Sütun Adı": [TURKISH_COLUMNS[name] for name in features],
                "Model Alanı": features,
            }),
            width="stretch",
            hide_index=True,
        )
    with st.expander("CSV Dosyası Nasıl Hazırlanır?"):
        st.markdown(
            "1. **Türkçe CSV Şablonunu İndir** düğmesine tıklayın.\n"
            "2. Her satıra bir öğrencinin bilgilerini girin.\n"
            "3. Kategori alanları için kod rehberini kullanın.\n"
            "4. Dosyayı CSV biçiminde kaydedip aşağıdan yükleyin.\n\n"
            "İngilizce sütun adlarına sahip eski CSV dosyaları da desteklenir."
        )


def render_batch_prediction():
    st.markdown("## Toplu Öğrenci Tahmini")
    stage = st.segmented_control(
        "Tahmin Aşaması",
        options=["A", "B"],
        default="B",
        format_func=lambda value: (
            "Aşama A · Kayıt Dönemi" if value == "A"
            else "Aşama B · Birinci Dönem"
        ),
        key="batch_stage",
    )
    if stage not in {"A", "B"}:
        return

    try:
        engine, validator = load_batch_services()
    except Exception:
        st.error("Tahmin sistemi yüklenemedi. Model dosyalarını kontrol edin.")
        return
    features = list(engine.get_required_features(stage))

    try:
        show_csv_resources(features, validator, stage)
    except (ValueError, KeyError) as exc:
        st.error(f"CSV şablonu hazırlanamadı: {exc}")
        return

    st.markdown("### 2. CSV Dosyası Yükle")
    uploaded_file = st.file_uploader(
        "Öğrenci verilerini içeren CSV dosyası",
        type=["csv"],
        key=f"upload_batch_{stage}",
    )
    if uploaded_file is None:
        return
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        st.error(f"Dosya boyutu {MAX_FILE_MB} MB sınırını aşıyor.")
        return
    if not raw.strip():
        st.error("Yüklenen CSV dosyası boş.")
        return

    try:
        df = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"CSV dosyası okunamadı: {exc}")
        return
    if df.empty:
        st.error("CSV dosyasında öğrenci kaydı yok.")
        return
    if len(df) > MAX_ROWS:
        st.error(f"En fazla {MAX_ROWS} öğrenci yükleyebilirsiniz.")
        return

    try:
        df = normalize_uploaded_csv(df, features)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"CSV sütun hatası: {exc}")
        return

    st.success(f"{len(df)} öğrenci kaydı yüklendi.")
    st.markdown("### 3. Veri Önizleme")
    st.dataframe(
        create_turkish_preview(df, features).head(10),
        width="stretch",
        hide_index=True,
    )
    file_signature = (stage, sha256(raw).hexdigest())

    st.markdown("### 4. Toplu Tahmin")
    if st.button(
        "Toplu Tahmini Başlat",
        type="primary",
        width="stretch",
        key=f"run_batch_{stage}",
    ):
        try:
            results, errors = process_batch(
                df=df, stage=stage, engine=engine, validator=validator
            )
            st.session_state["batch_output"] = {
                "signature": file_signature,
                "results": results,
                "errors": errors,
                "stage": stage,
            }
        except Exception:
            st.error("Toplu tahmin sırasında beklenmeyen bir hata oluştu.")
            return

    saved = st.session_state.get("batch_output")
    if saved is None or saved["signature"] != file_signature:
        return
    results, errors = saved["results"], saved["errors"]
    show_summary(results, errors, stage)

    if not errors.empty:
        st.markdown("### Hatalı Kayıtlar")
        st.dataframe(errors, width="stretch", hide_index=True)
        st.download_button(
            "Hata Raporunu CSV Olarak İndir",
            data=errors.to_csv(index=False).encode("utf-8-sig"),
            file_name="academicrisk_hatalar.csv",
            mime="text/csv",
            width="stretch",
            key=f"download_errors_{stage}",
        )
