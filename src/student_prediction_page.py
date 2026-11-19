
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.prediction_engine import PredictionEngine
from src.input_validator import StudentInputValidator
from src.shap_engine import ShapEngine


# ==================================================
# UCI CATEGORY LABELS - TURKISH
# ==================================================

COURSES = {
    33: "Biyoyakıt Üretim Teknolojileri",
    171: "Animasyon ve Multimedya Tasarımı",
    8014: "Sosyal Hizmet - Akşam Öğretimi",
    9003: "Tarım Bilimleri (Agronomi)",
    9070: "İletişim Tasarımı",
    9085: "Veteriner Hemşireliği",
    9119: "Bilişim Mühendisliği",
    9130: "At Yetiştiriciliği",
    9147: "İşletme",
    9238: "Sosyal Hizmet",
    9254: "Turizm",
    9500: "Hemşirelik",
    9556: "Ağız Hijyeni",
    9670: "Reklamcılık ve Pazarlama Yönetimi",
    9773: "Gazetecilik ve İletişim",
    9853: "Temel Eğitim",
    9991: "İşletme - Akşam Öğretimi",
}

APPLICATION_MODES = {
    1: "1. Aşama - Genel Kontenjan",
    2: "612/93 Sayılı Düzenleme",
    5: "1. Aşama - Azor Adaları Özel Kontenjanı",
    7: "Başka Bir Yükseköğretim Programı Mezunu",
    10: "854-B/99 Sayılı Düzenleme",
    15: "Uluslararası Lisans Öğrencisi",
    16: "1. Aşama - Madeira Özel Kontenjanı",
    17: "2. Aşama - Genel Kontenjan",
    18: "3. Aşama - Genel Kontenjan",
    26: "533-A/99 Düzenlemesi - Farklı Plan",
    27: "533-A/99 Düzenlemesi - Başka Kurum",
    39: "23 Yaş Üstü Aday",
    42: "Yatay Geçiş",
    43: "Program Değişikliği",
    44: "Teknolojik Uzmanlık Diploması",
    51: "Kurum veya Program Değişikliği",
    53: "Kısa Dönem Yükseköğretim Diploması",
    57: "Uluslararası Kurum veya Program Değişikliği",
}

QUALIFICATIONS = {
    1: "Ortaöğretim / Lise",
    2: "Yükseköğretim - Lisans Diploması",
    3: "Yükseköğretim - Üniversite Derecesi",
    4: "Yükseköğretim - Yüksek Lisans",
    5: "Yükseköğretim - Doktora",
    6: "Yükseköğretime Devam Etmiş",
    9: "12. Sınıf - Tamamlanmamış",
    10: "11. Sınıf - Tamamlanmamış",
    12: "Diğer - 11. Sınıf",
    14: "10. Sınıf",
    15: "10. Sınıf - Tamamlanmamış",
    19: "Temel Eğitim - 3. Kademe",
    38: "Temel Eğitim - 2. Kademe",
    39: "Teknolojik Uzmanlık Eğitimi",
    40: "Yükseköğretim - 1. Kademe",
    42: "Mesleki Yüksek Teknik Program",
    43: "Yükseköğretim - 2. Kademe Yüksek Lisans",
}

CATEGORY_MAPS = {
    "Application mode": APPLICATION_MODES,
    "Course": COURSES,
    "Previous qualification": QUALIFICATIONS,
}

LABELS = {
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

STATUS = {
    "Graduate": "Mezuniyet",
    "Enrolled": "Öğrenime Devam",
    "Dropout": "Okuldan Ayrılma",
}

GROUPS = {
    "Kayıt ve Program Bilgileri": [
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Age at enrollment",
    ],
    "Önceki Eğitim ve Giriş Bilgileri": [
        "Previous qualification",
        "Previous qualification (grade)",
        "Admission grade",
    ],
    "Birinci Dönem Akademik Bilgileri": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
}


# ==================================================
# ENGINE
# ==================================================

@st.cache_resource
def load_services():
    engine = PredictionEngine()
    validator = StudentInputValidator(engine)
    shap_engine = ShapEngine(engine)

    return engine, validator, shap_engine


def percent(value):
    return f"%{float(value) * 100:.2f}".replace(".", ",")


# ==================================================
# INPUT FIELD RENDERING
# ==================================================

def category_field(feature, allowed, stage):
    mapping = CATEGORY_MAPS[feature]

    values = sorted(
        int(x) for x in allowed[feature]
    )

    unknown_labels = [
        value for value in values
        if value not in mapping
    ]

    if unknown_labels:
        raise ValueError(
            f"{feature} için açıklaması olmayan "
            f"kodlar: {unknown_labels}"
        )

    return st.selectbox(
        LABELS[feature],
        options=values,
        index=None,
        placeholder="Seçiniz...",
        format_func=lambda code: (
            f"{mapping[code]} ({code})"
        ),
        key=f"student_{stage}_{feature}",
    )


def number_field(
    feature,
    stage,
    minimum,
    maximum,
    integer=True,
):
    key = f"student_{stage}_{feature}"

    if integer:
        return st.number_input(
            LABELS[feature],
            min_value=int(minimum),
            max_value=(
                int(maximum)
                if maximum is not None else None
            ),
            value=None,
            step=1,
            placeholder="Değer giriniz",
            key=key,
        )

    return st.number_input(
        LABELS[feature],
        min_value=float(minimum),
        max_value=float(maximum),
        value=None,
        step=0.1,
        format="%.2f",
        placeholder="Değer giriniz",
        key=key,
    )


def render_field(feature, allowed, stage):

    if feature in CATEGORY_MAPS:
        return category_field(feature, allowed, stage)

    if feature == "Daytime/evening attendance":
        return st.selectbox(
            LABELS[feature],
            options=[1, 0],
            index=None,
            placeholder="Seçiniz...",
            format_func=lambda x: (
                "Gündüz Öğretimi" if x == 1
                else "Akşam Öğretimi"
            ),
            key=f"student_{stage}_{feature}",
        )

    if feature == "Age at enrollment":
        return number_field(
            feature, stage, 1, 120
        )

    if feature == "Application order":
        return number_field(
            feature, stage, 0, 9
        )

    if feature in {
        "Previous qualification (grade)",
        "Admission grade",
    }:
        return number_field(
            feature, stage, 0, 200,
            integer=False,
        )

    if feature == "Curricular units 1st sem (grade)":
        return number_field(
            feature, stage, 0, 20,
            integer=False,
        )

    if feature.startswith("Curricular units 1st sem"):
        return number_field(
            feature, stage, 0, None
        )

    raise ValueError(
        f"Tanınmayan özellik: {feature}"
    )


# ==================================================
# FORM SECTIONS
# ==================================================

def form_section(title, section_features, allowed, stage):

    st.markdown(f"#### {title}")

    left, right = st.columns(2, gap="large")

    values = {}

    for index, feature in enumerate(section_features):
        col = left if index % 2 == 0 else right

        with col:
            values[feature] = render_field(
                feature, allowed, stage
            )

    return values


# ==================================================
# RESULT DISPLAY
# ==================================================

def render_results(result, stage):
    """Show predictions without changing the engine's probabilities."""
    st.markdown("### Tahmin Sonuçları")

    predicted_code = result["predicted_status"]
    predicted = STATUS.get(predicted_code, predicted_code)
    st.success(f"Tahmin Edilen Durum: {predicted}")

    probabilities = result["status_probabilities"]
    keys = ("Graduate", "Enrolled", "Dropout")
    columns = st.columns(3)
    for column, key in zip(columns, keys):
        with column:
            st.metric(STATUS[key], percent(probabilities[key]))

    chart = go.Figure(
        go.Bar(
            x=[float(probabilities[key]) for key in keys],
            y=[STATUS[key] for key in keys],
            orientation="h",
            marker_color=["#6658DB", "#3BA6C7", "#E5A14A"],
            text=[percent(probabilities[key]) for key in keys],
            textposition="auto",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    chart.update_layout(
        template="plotly_white",
        height=245,
        margin=dict(l=10, r=30, t=10, b=25),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, width="stretch")

    if stage == "B":
        risk = float(result["dropout_risk_score"])
        threshold = float(result["threshold"])
        st.markdown("### Erken Uyarı")
        left, right = st.columns(2)
        with left:
            st.metric("Risk Skoru", percent(risk))
        with right:
            st.metric("Uyarı Eşiği", percent(threshold))
        st.progress(min(max(risk, 0.0), 1.0))
        if result["warning"]:
            st.warning("Erken uyarı oluşturuldu.")
        else:
            st.success("Erken uyarı oluşturulmadı.")



# ==================================================
# SHAP EXPLANATION UI
# ==================================================

def readable_shap_value(feature, value):
    """Render a validated input value in Turkish without altering the model."""
    if pd.isna(value):
        return "-"

    if feature in CATEGORY_MAPS:
        code = int(value)
        return CATEGORY_MAPS[feature].get(code, str(code))

    if feature == "Daytime/evening attendance":
        return "Gündüz Öğretimi" if int(value) == 1 else "Akşam Öğretimi"

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}".replace(".", ",")
    except (ValueError, TypeError, OverflowError):
        return str(value)


def prepare_shap_display(shap_result):
    """Prepare UI-only representations; preserve SHAP contributions exactly."""
    contributions = shap_result["contributions"].copy(deep=True)
    required_columns = {
        "feature", "label", "original_value", "shap_log_odds", "abs_shap"
    }
    if not required_columns.issubset(contributions.columns):
        raise ValueError("SHAP sonuçlarında gerekli alanlar eksik.")

    values = pd.to_numeric(contributions["shap_log_odds"], errors="raise")
    if values.isna().any():
        raise ValueError("SHAP katkılarında geçersiz değer var.")

    contributions = contributions.assign(
        shap_log_odds=values,
        abs_shap=values.abs(),
    ).sort_values("abs_shap", ascending=False, kind="stable").reset_index(drop=True)

    contributions["Girilen Değer"] = [
        readable_shap_value(row.feature, row.original_value)
        for row in contributions.itertuples(index=False)
    ]
    contributions["Etki Yönü"] = contributions["shap_log_odds"].map(
        lambda value: "Artırıyor" if value > 0
        else "Azaltıyor" if value < 0 else "Nötr"
    )
    contributions["Katkı Metni"] = contributions["shap_log_odds"].map(
        lambda value: f"{value:+.4f}" if value != 0 else "0.0000"
    )
    return contributions


def render_shap_explanation(shap_result):
    """Show Stage B SHAP values clearly, without re-computing predictions."""
    st.markdown("### Risk Skorunu Etkileyen Özellikler")

    contributions = prepare_shap_display(shap_result)
    top = contributions.head(8).sort_values("shap_log_odds", kind="stable")

    chart = go.Figure(
        go.Bar(
            x=top["shap_log_odds"].astype(float).tolist(),
            y=top["label"].astype(str).tolist(),
            orientation="h",
            marker_color=[
                "#D96B75" if value > 0 else "#199B88"
                for value in top["shap_log_odds"]
            ],
            customdata=[[value] for value in top["Girilen Değer"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP katkısı: %{x:+.4f}<br>"
                "Girilen değer: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    chart.add_vline(x=0, line_width=1, line_color="#8A8FA3")
    chart.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=5, r=25, t=10, b=45),
        xaxis_title="SHAP Katkısı (Log-Odds)",
        xaxis=dict(zeroline=False, showgrid=True),
        yaxis=dict(title="", automargin=True, tickfont=dict(size=12)),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, width="stretch", key="student_shap_chart")

    left, right = st.columns(2, gap="large")
    increasing = contributions.loc[contributions["shap_log_odds"] > 0].head(5)
    decreasing = contributions.loc[contributions["shap_log_odds"] < 0].head(5)

    with left:
        st.markdown("**Riski Artıran Özellikler**")
        if increasing.empty:
            st.write("Bu tahminde riski artıran katkı bulunamadı.")
        else:
            for row in increasing.itertuples(index=False):
                st.markdown(
                    f"**{row.label}**  ·  "
                    f":red[{row.shap_log_odds:+.4f}]"
                )

    with right:
        st.markdown("**Riski Azaltan Özellikler**")
        if decreasing.empty:
            st.write("Bu tahminde riski azaltan katkı bulunamadı.")
        else:
            for row in decreasing.itertuples(index=False):
                st.markdown(
                    f"**{row.label}**  ·  "
                    f":green[{row.shap_log_odds:+.4f}]"
                )

    with st.expander("Tüm Özelliklerin Katkılarını Gör", expanded=False):
        detail = contributions.rename(
            columns={"label": "Özellik", "shap_log_odds": "SHAP Katkısı"}
        )
        st.dataframe(
            detail[["Özellik", "Girilen Değer", "SHAP Katkısı", "Etki Yönü"]],
            width="stretch",
            height=min(590, 40 + len(detail) * 36),
            hide_index=True,
            column_config={
                "Özellik": st.column_config.TextColumn("Özellik", width="large"),
                "Girilen Değer": st.column_config.TextColumn("Girilen Değer", width="medium"),
                "SHAP Katkısı": st.column_config.NumberColumn(
                    "SHAP Katkısı", format="%.4f", width="small"
                ),
                "Etki Yönü": st.column_config.TextColumn("Etki Yönü", width="small"),
            },
        )


# ==================================================
# PAGE
# ==================================================

def render_student_prediction():

    st.markdown("## Öğrenci Tahmini")

    engine, validator, shap_engine = load_services()

    stage = st.segmented_control(
        "Tahmin Aşaması",
        options=["A", "B"],
        default="A",
        format_func=lambda x: (
            "Aşama A · Kayıt Dönemi"
            if x == "A"
            else "Aşama B · Birinci Dönem"
        ),
        key="student_stage",
    )

    if stage not in {"A", "B"}:
        return

    required = engine.get_required_features(stage)
    allowed = validator.get_allowed_categories(stage)

    st.markdown("---")

    with st.form(f"student_form_{stage}"):

        inputs = {}

        for index, (title, feature_names) in enumerate(
            GROUPS.items(),
            start=1,
        ):
            available = [
                feature
                for feature in feature_names
                if feature in required
            ]

            if not available:
                continue

            st.caption(f"{index:02d} / ÖĞRENCİ BİLGİLERİ")

            inputs.update(
                form_section(
                    title,
                    available,
                    allowed,
                    stage,
                )
            )

            st.markdown("")

        st.markdown("---")

        submitted = st.form_submit_button(
            "Tahmin Et",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return

    missing = [
        LABELS.get(feature, feature)
        for feature, value in inputs.items()
        if value is None
    ]

    if missing:
        st.error(
            "Lütfen tüm alanları doldurun: "
            + ", ".join(missing)
        )
        return

    try:
        clean_data = validator.validate(
            inputs,
            stage=stage,
        )

        result = engine.predict_one(
            clean_data.iloc[0].to_dict(),
            stage=stage,
        )

    except (ValueError, TypeError, KeyError) as exc:
        st.error(
            f"Girdi veya tahmin hatası: {exc}"
        )
        return

    render_results(result, stage)

    # SHAP explanations are available only for Stage B.

    if stage == "B":

        try:
            shap_result = shap_engine.explain_one(
                clean_data.iloc[0].to_dict()
            )

            predicted_risk = float(
                result["dropout_risk_score"]
            )

            explained_risk = float(
                shap_result["dropout_probability"]
            )

            if abs(
                predicted_risk - explained_risk
            ) > 1e-8:

                raise ValueError(
                    "Tahmin ve SHAP risk skorları "
                    "birbiriyle eşleşmiyor."
                )

            render_shap_explanation(
                shap_result
            )

        except (ValueError, TypeError, KeyError) as exc:

            st.error(
                f"SHAP açıklaması oluşturulamadı: {exc}"
            )

