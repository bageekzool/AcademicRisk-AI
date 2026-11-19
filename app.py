"""AcademicRisk AI: redesigned Turkish Streamlit dashboard.

UI-only redesign. Predictions and SHAP stay in the existing src modules.
"""
from pathlib import Path
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.student_prediction_page import render_student_prediction
from src.batch_prediction_page import render_batch_prediction

ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "reports" / "final_test"
RESULTS_FILE = FINAL_DIR / "final_test_results.csv"

PURPLE = "#7257EA"
NAVY = "#10142D"
TEAL = "#0E9E9A"
CORAL = "#E36D7B"

st.set_page_config(
    page_title="AcademicRisk AI | Akademik Risk Analiz Sistemi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], button, input {font-family:Inter, sans-serif;}
.stApp {background: #F4F5FC; color: #1C2440;}
[data-testid="stSidebar"] {display:none;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1380px; padding-top:1.0rem; padding-bottom:2rem;}
#MainMenu {visibility:hidden;}
.brand-line {display:flex;align-items:center;gap:13px;padding:9px 0 8px;}
.brand-logo {width:47px;height:47px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,#7865EC,#3F36A9);border-radius:14px;color:white;font-size:24px;
 box-shadow:0 8px 18px rgba(90,70,187,.2);}
.brand-name {font-size:1.42rem;font-weight:800;letter-spacing:-.045em;color:#171C36;line-height:1.2;}
.brand-sub {color:#858CA6;font-size:.76rem;font-weight:500;margin-top:3px;}
.nav-shell {border-bottom:1px solid #E0E4EF;margin:3px 0 12px;}
/* Navigation remains native Streamlit buttons for accessibility. */
.stButton > button {border-radius:12px!important;font-weight:650!important;
 border:1px solid #E1E4F0!important;background:#FFF;color:#47516D;
 min-height:43px;transition:filter .15s, box-shadow .15s;}
.stButton > button:hover {border-color:#B6A8FA!important;box-shadow:0 4px 15px #473db016;}
.stButton > button[kind="primary"] {background:#6C55D7!important;color:#FFF!important;
 border-color:#6C55D7!important;box-shadow:0 5px 13px #6b55d730;}
.stButton > button:focus-visible {outline:3px solid #B5A7FF;outline-offset:2px;}
.hero-new {position:relative;overflow:hidden;border-radius:26px;
 background:linear-gradient(118deg,#131932 0%,#26205D 53%,#614CD0 100%);
 box-shadow:0 18px 40px #2f246728; padding:42px 45px;color:#FFF;margin:17px 0 26px;}
.hero-new::before {content:'';position:absolute;right:-110px;top:-160px;
 width:520px;height:520px;border-radius:50%;border:90px solid #ffffff12;}
.hero-new::after {content:'';position:absolute;right:11%;bottom:-195px;
 width:340px;height:340px;border-radius:50%;background:#957cfa28;}
.hero-content {position:relative;z-index:1;max-width:750px;}
.eyebrow {display:inline-block;letter-spacing:.12em;color:#C7B8FF;font-size:.73rem;font-weight:800;margin-bottom:16px;}
.hero-new h1 {color:white!important;font-size:clamp(2rem,3.6vw,3.25rem)!important;line-height:1.18!important;
 letter-spacing:-.055em!important;font-weight:800!important;margin:0 0 15px!important;}
.hero-new p {color:#DCD8F8;font-size:.95rem;max-width:650px;line-height:1.78;margin:0;}
.hero-tags {display:flex;flex-wrap:wrap;gap:9px;margin-top:23px;}
.hero-tag {background:#ffffff18;border:1px solid #ffffff29;border-radius:999px;color:#F2EFFF;
 padding:7px 12px;font-size:.77rem;font-weight:600;}
.section-head {margin:22px 0 14px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;}
.section-head h2 {font-size:1.25rem!important;color:#202641!important;letter-spacing:-.025em;
 font-weight:800!important;margin:0!important;}
.section-head span {font-size:.79rem;color:#8B92A7;}
.kpi-new {position:relative;background:#fff;border-radius:18px;padding:24px 22px 21px;
 border:1px solid #E9EAF3;box-shadow:0 7px 24px #21265e09;min-height:165px;overflow:hidden;}
.kpi-new::before {content:'';position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent);}
.kpi-top {display:flex;justify-content:space-between;gap:8px;align-items:center;}
.kpi-label-new {color:#778098;font-size:.79rem;font-weight:650;}
.kpi-symbol {display:flex;align-items:center;justify-content:center;border-radius:10px;
 width:33px;height:33px;background:var(--soft);color:var(--accent);font-size:18px;}
.kpi-number {font-size:2.1rem;font-weight:800;letter-spacing:-.06em;color:#242845;margin:15px 0 5px;line-height:1.1;}
.kpi-foot {color:#9298AC;font-size:.73rem;}
.panel {background:white;border:1px solid #E8EAF4;border-radius:18px;padding:17px 19px 7px;
 box-shadow:0 7px 23px #24296308;}
.panel-label {font-size:1.01rem;color:#252B46;font-weight:750;margin:2px 0 7px;}
.panel-sub {font-size:.75rem;color:#8B92A7;margin-bottom:6px;}
.feature-new {background:white;border:1px solid #E8EAF3;border-radius:18px;padding:23px 22px;
 box-shadow:0 7px 22px #20275b07;min-height:158px;}
.feature-icon {font-size:23px;margin-bottom:13px;}
.feature-title {font-size:.98rem;font-weight:750;color:#272C48;margin-bottom:7px;}
.feature-description {font-size:.79rem;color:#7F879D;line-height:1.65;}
.stage-card {background:#FFF;border-radius:18px;border:1px solid #E8EAF3;padding:26px 24px;margin:4px 0 8px;min-height:172px;}
.stage-id {font-size:.70rem;color:#7C63E2;letter-spacing:.12em;font-weight:800;}
.stage-title {color:#202645;font-size:1.13rem;font-weight:800;margin-top:11px;}
.stage-text {color:#80879C;font-size:.83rem;line-height:1.65;margin-top:7px;}
.page-banner {background:linear-gradient(105deg,#161C3A,#5144A2);border-radius:20px;padding:26px 29px;
 margin:18px 0 22px;color:#FFF;}
.page-banner h1 {color:white!important;margin:3px 0 6px!important;font-size:1.9rem!important;font-weight:800!important;}
.page-banner p {color:#D9D6F3;font-size:.89rem;margin:0;}
.data-note {font-size:.8rem;color:#737E98;border-left:3px solid #A79AEC;padding:7px 12px;margin:16px 0;}
.footer-new {border-top:1px solid #DFE2EF;margin-top:36px;padding:20px 0 8px;
 text-align:center;color:#8B91A8;font-size:.77rem;}
[data-testid="stPlotlyChart"] {background:transparent;border:0;padding:0;}
@media(max-width:760px){.block-container{padding-left:.9rem;padding-right:.9rem;}
 .hero-new{padding:30px 24px;}.hero-new h1{font-size:1.9rem!important;}
 .kpi-new{min-height:140px;}.kpi-number{font-size:1.75rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Sonuç dosyası bulunamadı: {RESULTS_FILE}")
    data = pd.read_csv(RESULTS_FILE)
    required = {"multiclass_A", "multiclass_B", "binary_warning_B"}
    if (
        "model" not in data.columns
        or len(data) != 3
        or set(data["model"]) != required
        or data["model"].duplicated().any()
    ):
        raise ValueError("Kaydedilen model sonuçları beklenen yapıda değil.")
    return data.set_index("model")


@st.cache_data
def load_matrix(model):
    allowed = {"multiclass_A", "multiclass_B", "binary_warning_B"}
    if model not in allowed:
        raise ValueError("Geçersiz model adı.")
    path = FINAL_DIR / f"{model}_confusion_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def metric(results, model, column):
    return float(results.loc[model, column])


def pct(value):
    return f"%{value * 100:.2f}".replace(".", ",")


def decimal(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def section(title, description=""):
    st.markdown(
        f'<div class="section-head"><h2>{escape(title)}</h2>'
        f'<span>{escape(description)}</span></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, note="", accent="#7257EA", soft="#F0ECFF", symbol="✦"):
    st.markdown(
        '<div class="kpi-new" style="--accent:' + accent + ';--soft:' + soft + '">'
        '<div class="kpi-top"><div class="kpi-label-new">' + escape(str(label)) + '</div>'
        '<div class="kpi-symbol">' + escape(str(symbol)) + '</div></div>'
        '<div class="kpi-number">' + escape(str(value)) + '</div>'
        '<div class="kpi-foot">' + escape(str(note)) + '</div></div>',
        unsafe_allow_html=True,
    )


def info_card(icon, title, description):
    st.markdown(
        '<div class="feature-new"><div class="feature-icon">' + escape(str(icon)) + '</div>'
        '<div class="feature-title">' + escape(title) + '</div>'
        '<div class="feature-description">' + escape(description) + '</div></div>',
        unsafe_allow_html=True,
    )


def page_banner(title, description):
    st.markdown(
        '<div class="page-banner"><h1>' + escape(title) + '</h1><p>'
        + escape(description) + '</p></div>', unsafe_allow_html=True,
    )


def plot_style(fig, height=350):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=12, r=18, t=25, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#5D6580", size=12),
        legend=dict(orientation="h", y=-0.20),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#F0F1F7", zeroline=False)
    return fig


def performance_chart(results):
    labels = ["Aşama A", "Aşama B"]
    accuracy = [metric(results, "multiclass_A", "accuracy"),
                metric(results, "multiclass_B", "accuracy")]
    f1 = [metric(results, "multiclass_A", "macro_f1"),
          metric(results, "multiclass_B", "macro_f1")]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Doğruluk", x=labels, y=accuracy,
                         marker_color=PURPLE, text=[pct(x) for x in accuracy],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Macro F1", x=labels, y=f1,
                         marker_color=TEAL, text=[pct(x) for x in f1],
                         textposition="outside"))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.15],
                        tickformat=".0%"), bargap=.34, showlegend=True)
    return plot_style(fig)


def warning_chart(results):
    item = results.loc["binary_warning_B"]
    names = ["Doğru uyarı", "Yanlış uyarı", "Kaçırılan risk", "Doğru negatif"]
    counts = [int(item["tp"]), int(item["fp"]), int(item["fn"]), int(item["tn"])]
    colors = [TEAL, "#F0A64C", CORAL, "#6875D5"]
    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation="h", marker_color=colors,
        text=counts, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} öğrenci<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(title="Öğrenci sayısı", range=[0, max(counts + [1]) * 1.23]),
                      yaxis=dict(autorange="reversed"))
    return plot_style(fig)


def matrix_chart(matrix, title):
    if matrix is None:
        st.warning("Karmaşıklık matrisi bulunamadı.")
        return
    label_map = {"Dropout": "Ayrılma", "Enrolled": "Devam",
                 "Graduate": "Mezun", "Not Dropout": "Ayrılmama"}
    x = [label_map.get(str(v), str(v)) for v in matrix.columns]
    y = [label_map.get(str(v), str(v)) for v in matrix.index]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=x, y=y, text=matrix.values,
        texttemplate="%{text}",
        colorscale=[[0, "#F4F1FF"], [1, PURPLE]],
        hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Sayı: %{z}<extra></extra>",
        colorbar=dict(thickness=10),
    ))
    fig.update_layout(xaxis_title="Tahmin Edilen", yaxis_title="Gerçek",
                      yaxis=dict(autorange="reversed"))
    st.markdown(f'<div class="panel-label">{escape(title)}</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_style(fig, 355), width="stretch")


def panel_chart(title, subtitle, fig):
    st.markdown(f'<div class="panel-label">{escape(title)}</div>'
                f'<div class="panel-sub">{escape(subtitle)}</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch")


# Header and navigation (all six original pages preserved)
st.markdown('''<div class="brand-line"><div class="brand-logo">🎓</div>
<div><div class="brand-name">AcademicRisk AI</div>
<div class="brand-sub">Akademik Tahmin ve Erken Uyarı Platformu</div></div></div>
<div class="nav-shell"></div>''', unsafe_allow_html=True)

pages = ["Ana Sayfa", "Öğrenci Tahmini", "Toplu Tahmin",
         "Model Performansı", "Erken Uyarı", "Sistem Hakkında"]
if st.session_state.get("active_page") not in pages:
    st.session_state.active_page = "Ana Sayfa"

nav_cols = st.columns(6, gap="small")
for col, page_name in zip(nav_cols, pages):
    with col:
        if st.button(page_name, key=f"nav_{page_name}", width="stretch",
                     type="primary" if st.session_state.active_page == page_name
                     else "secondary"):
            st.session_state.active_page = page_name
            st.rerun()
page = st.session_state.active_page

try:
    results = load_results()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Veriler yüklenemedi: {exc}")
    st.stop()

if page == "Ana Sayfa":
    st.markdown('''<div class="hero-new"><div class="hero-content">
    <span class="eyebrow">VERİDEN İÇGÖRÜYE · YAPAY ZEKÂ DESTEKLİ</span>
    <h1>Akademik başarıyı öngörün.<br>Riskleri erken fark edin.</h1>
    <p>Öğrencilerin akademik durumlarını iki farklı aşamada tahmin edin,
    birinci dönem sonrası risk skorlarını inceleyin ve model kararlarını SHAP ile keşfedin.</p>
    <div class="hero-tags"><span class="hero-tag">🎯 3 sınıflı tahmin</span>
    <span class="hero-tag">🚦 Erken uyarı</span>
    <span class="hero-tag">✧ Açıklanabilir yapay zekâ</span></div>
    </div></div>''', unsafe_allow_html=True)

    section("Genel Bakış", "Bağımsız test sonuçları")
    a, b, c, d = st.columns(4, gap="medium")
    with a:
        kpi_card("Öğrenci kaydı", "4.424", "UCI veri kümesi", PURPLE, "#EEEAFE", "▦")
    with b:
        kpi_card("Aşama B · Macro F1", pct(metric(results, "multiclass_B", "macro_f1")),
                 "Üç sınıflı tahmin", TEAL, "#E4F8F6", "↗")
    with c:
        kpi_card("Ayrılma · Recall", pct(metric(results, "binary_warning_B", "dropout_recall")),
                 "Risk tespit duyarlılığı", CORAL, "#FDEEF0", "◉")
    with d:
        kpi_card("ROC-AUC", pct(metric(results, "binary_warning_B", "roc_auc")),
                 "İkili uyarı modeli", "#4D8FD6", "#EAF3FF", "◈")

    section("Model Analizleri", "İki aşamalı tahmin ve risk uyarıları")
    left, right = st.columns(2, gap="large")
    with left:
        panel_chart("Akademik Tahmin Performansı", "Aşama A ve B · Accuracy / Macro F1 (%)", performance_chart(results))
    with right:
        panel_chart("Erken Uyarı Sonuçları", "Test kümesindeki uyarı dağılımı", warning_chart(results))

    section("Nasıl Çalışır?", "İki aşamalı değerlendirme")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('''<div class="stage-card"><div class="stage-id">01 · AŞAMA A</div>
        <div class="stage-title">Kayıt dönemi tahmini</div>
        <div class="stage-text">Kayıt sırasında bilinen 8 özellik ile öğrencinin
        akademik durumu tahmin edilir.</div></div>''', unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="stage-card"><div class="stage-id">02 · AŞAMA B</div>
        <div class="stage-title">Birinci dönem ve erken uyarı</div>
        <div class="stage-text">İlk dönem akademik sonuçları dahil 14 özellik ile
        durum tahmini ve ayrı ikili modelle risk değerlendirmesi yapılır.</div></div>''', unsafe_allow_html=True)

    section("Sistem Özellikleri")
    a, b, c = st.columns(3, gap="medium")
    with a:
        info_card("👤", "Bireysel Tahmin", "Bir öğrenci için akademik durum ve erken uyarı analizi.")
    with b:
        info_card("📁", "Türkçe Toplu CSV", "Çoklu öğrenci analizi ve Türkçe sonuç dosyası.")
    with c:
        info_card("🧠", "SHAP Açıklamaları", "Risk skoruna katkıda bulunan model özellikleri.")
    st.markdown('<div class="data-note">Sonuçlar Portekiz kaynaklı UCI veri kümesinin ayrılmış '
                'test bölümüne aittir; canlı üniversite istatistiği değildir.</div>',
                unsafe_allow_html=True)

elif page == "Öğrenci Tahmini":
    render_student_prediction()

elif page == "Toplu Tahmin":
    render_batch_prediction()

elif page == "Model Performansı":
    page_banner("Model Performansı", "Seçilen modellerin bağımsız test kümesindeki sonuçları")
    a, b, c = st.columns(3, gap="medium")
    with a:
        kpi_card("Aşama A · Macro F1", pct(metric(results, "multiclass_A", "macro_f1")),
                 "Kayıt dönemi", PURPLE, "#EEEAFE", "A")
    with b:
        kpi_card("Aşama B · Macro F1", pct(metric(results, "multiclass_B", "macro_f1")),
                 "Birinci dönem", TEAL, "#E4F8F6", "B")
    with c:
        kpi_card("Aşama B · Doğruluk", pct(metric(results, "multiclass_B", "accuracy")),
                 "Üç sınıflı model", "#4D8FD6", "#EAF3FF", "%")
    section("Performans Karşılaştırması")
    panel_chart("Accuracy ve Macro F1", "Aynı test kümesi · skorlar yüzde (%) biçiminde", performance_chart(results))
    section("Karmaşıklık Matrisleri", "Gerçek ve tahmin edilen öğrenci durumları")
    left, right = st.columns(2, gap="large")
    with left:
        matrix_chart(load_matrix("multiclass_A"), "Aşama A")
    with right:
        matrix_chart(load_matrix("multiclass_B"), "Aşama B")

elif page == "Erken Uyarı":
    page_banner("Erken Uyarı Analizi", "Birinci dönem verileriyle çalışan ikili risk modelinin test sonuçları")
    m = "binary_warning_B"
    a, b, c, d = st.columns(4, gap="medium")
    with a:
        kpi_card("Recall", pct(metric(results, m, "dropout_recall")),
                 "Gerçek ayrılmaların tespiti", TEAL, "#E4F8F6", "↗")
    with b:
        kpi_card("Precision", pct(metric(results, m, "dropout_precision")),
                 "Uyarıların isabet oranı", PURPLE, "#EEEAFE", "◎")
    with c:
        kpi_card("F2 Skoru", pct(metric(results, m, "dropout_f2")),
                 "Recall ağırlıklı skor", "#4D8FD6", "#EAF3FF", "F₂")
    with d:
        kpi_card("Karar Eşiği", pct(metric(results, m, "threshold")),
                 "Sabit uyarı sınırı", CORAL, "#FDEEF0", "!")
    section("Uyarı Dağılımı ve Karmaşıklık Matrisi")
    left, right = st.columns(2, gap="large")
    with left:
        panel_chart("Risk Uyarılarının Dağılımı", "Doğru, yanlış ve kaçırılan tahminler", warning_chart(results))
    with right:
        matrix_chart(load_matrix(m), "İkili Model · Confusion Matrix")
    st.markdown('<div class="data-note">Uyarı, öğrencinin kesinlikle okuldan ayrılacağını '
                'göstermez. Sonuçlar akademik değerlendirme için destekleyici bilgidir.</div>',
                unsafe_allow_html=True)

elif page == "Sistem Hakkında":
    page_banner("Sistem Hakkında", "AcademicRisk AI araştırma prototipinin kapsamı ve bileşenleri")
    section("Modelin İşleyişi", "İki aşamalı akademik değerlendirme")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('''<div class="stage-card"><div class="stage-id">AŞAMA A · 8 ÖZELLİK</div>
        <div class="stage-title">Kayıt Dönemi</div><div class="stage-text">Öğrencinin kayıt bilgileri
        üzerinden Graduate, Enrolled ve Dropout durumu tahmin edilir.</div></div>''',
        unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="stage-card"><div class="stage-id">AŞAMA B · 14 ÖZELLİK</div>
        <div class="stage-title">Birinci Dönem</div><div class="stage-text">Dönem bilgileri eklenerek
        akademik durum ve ayrı ikili modelle erken uyarı skoru hesaplanır.</div></div>''',
        unsafe_allow_html=True)
    section("Mevcut Modüller")
    a, b, c = st.columns(3, gap="medium")
    with a:
        info_card("👤", "Öğrenci Tahmini", "Tek öğrenci girdisiyle durum tahmini ve SHAP açıklamaları.")
    with b:
        info_card("📂", "Toplu Tahmin", "Türkçe CSV ile toplu değerlendirme ve sonuç indirme.")
    with c:
        info_card("📊", "Model Performansı", "Kaydedilmiş test ölçümleri ve karmaşıklık matrisleri.")
    section("Kullanılan Teknolojiler")
    st.write("Python · Streamlit · Scikit-learn · XGBoost · SHAP · Pandas · NumPy · Plotly")
    section("Veri ve Kullanım Sınırları")
    st.markdown('<div class="data-note">Modeller, Portekiz’deki bir yükseköğretim '
                'kurumundan elde edilen veri kümesi üzerinde geliştirilmiştir. '
                'Diğer üniversitelerde geçerliliği ayrıca doğrulanmalıdır; '
                'sistem araştırma amaçlıdır.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-new">© 2026 AcademicRisk AI · Akademik Risk Analiz Sistemi</div>',
            unsafe_allow_html=True)
