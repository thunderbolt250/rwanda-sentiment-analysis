"""
Phase 6 — Streamlit Dashboard
Rwanda Political Sentiment Analysis
=====================================
Interactive dashboard to explore sentiment trends,
model predictions, and SHAP explainability.

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Rwanda Political Sentiment",
    page_icon="🇷🇼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    .positive { border-color: #2ECC71; }
    .negative { border-color: #E74C3C; }
    .neutral  { border-color: #95A5A6; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────
COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral" : "#95A5A6"
}

# ─────────────────────────────────────────
#  LOAD DATA & MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_labeled_tweets.csv")
    df = df.dropna(subset=["final_label", "clean_text"])
    df["final_label"] = df["final_label"].str.lower().str.strip()
    df = df[df["final_label"].isin(["positive", "neutral", "negative"])]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        df["date"]  = df["timestamp"].dt.date
    return df

@st.cache_resource
def load_models():
    lr    = joblib.load("models/logistic_regression.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return lr, tfidf

@st.cache_data
def load_shap():
    path = "results/explainability/shap_word_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s',!?.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────
df       = load_data()
lr, tfidf = load_models()
shap_df  = load_shap()

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("images/Flag_of_Rwanda.svg", width=100)
    st.markdown("## Rwanda Sentiment")
    st.markdown("---")

    # filters
    st.markdown("### Filters")
    selected_labels = st.multiselect(
        "Sentiment",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"]
    )

    if "query" in df.columns:
        queries = ["All"] + sorted(df["query"].dropna().unique().tolist())
        selected_query = st.selectbox("Query / Topic", queries)
    else:
        selected_query = "All"

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Project:** Rwanda Political Sentiment Analysis

    **Data:** ~692 tweets (2023–2024)

    **Models:**
    - Logistic Regression + TF-IDF
    - AfroXLMR (77.2% accuracy)

    **Built by:** Mwesigye Emmy
    CMU Africa
    """)

# ─────────────────────────────────────────
#  APPLY FILTERS
# ─────────────────────────────────────────
filtered = df[df["final_label"].isin(selected_labels)]
if selected_query != "All" and "query" in df.columns:
    filtered = filtered[filtered["query"] == selected_query]

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
    <h1 style="font-size:2.8rem; font-weight:800; color:#ffffff;
               text-shadow: 0 2px 8px rgba(0,0,0,0.4); margin-bottom:0.2rem;">
        Rwanda Political Sentiment Analysis
    </h1>
    <p style="font-size:1.1rem; color:#aaaaaa; margin-top:0;">
        Analyzing public sentiment on Twitter/X about Rwandan politics (2023–2024)
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────
#  TOP METRICS
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
total = len(filtered)
counts = filtered["final_label"].value_counts()

with col1:
    st.metric("Total Tweets", f"{total:,}")
with col2:
    pos = counts.get("positive", 0)
    st.metric("😊 Positive", f"{pos:,}", f"{pos/total*100:.1f}%" if total else "")
with col3:
    neg = counts.get("negative", 0)
    st.metric("😠 Negative", f"{neg:,}", f"{neg/total*100:.1f}%" if total else "")
with col4:
    neu = counts.get("neutral", 0)
    st.metric("😐 Neutral", f"{neu:,}", f"{neu/total*100:.1f}%" if total else "")

st.markdown("---")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Trends Over Time",
    "🔍 Explainability",
    "🤖 Live Predictor"
])


# ══════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=COLORS,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=13)
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Tweets per Sentiment")
        fig_bar = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=COLORS,
            labels={"x": "Sentiment", "y": "Count"},
            text=counts.values
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # tweets per query
    if "query" in filtered.columns:
        st.subheader("Sentiment by Search Query")
        query_sent = (filtered.groupby(["query", "final_label"])
                               .size().reset_index(name="count"))
        fig_grouped = px.bar(
            query_sent, x="query", y="count",
            color="final_label",
            color_discrete_map=COLORS,
            barmode="group",
            labels={"query": "Search Query", "count": "Tweets", "final_label": "Sentiment"}
        )
        fig_grouped.update_layout(margin=dict(t=20, b=100),
                                   xaxis_tickangle=-30)
        st.plotly_chart(fig_grouped, use_container_width=True)

    # sample tweets table
    st.subheader("Sample Tweets")
    sentiment_filter = st.selectbox("Filter by sentiment",
                                    ["All", "positive", "neutral", "negative"],
                                    key="table_filter")
    show_df = filtered if sentiment_filter == "All" else \
              filtered[filtered["final_label"] == sentiment_filter]

    display_cols = ["clean_text", "final_label", "query"]
    display_cols = [c for c in display_cols if c in show_df.columns]
    st.dataframe(
        show_df[display_cols].head(20).rename(columns={
            "clean_text"  : "Tweet",
            "final_label" : "Sentiment",
            "query"       : "Query"
        }),
        use_container_width=True,
        height=300
    )


# ══════════════════════════════════════════
#  TAB 2 — TRENDS OVER TIME
# ══════════════════════════════════════════
with tab2:
    st.subheader("Sentiment Trend Over Time")

    if "month" in filtered.columns and filtered["month"].notna().any():
        monthly = (filtered.groupby(["month", "final_label"])
                            .size().reset_index(name="count"))
        monthly = monthly.sort_values("month")

        fig_line = px.line(
            monthly, x="month", y="count",
            color="final_label",
            color_discrete_map=COLORS,
            markers=True,
            labels={"month": "Month", "count": "Tweets", "final_label": "Sentiment"}
        )
        fig_line.update_layout(
            xaxis_tickangle=-45,
            legend_title="Sentiment",
            hovermode="x unified"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # stacked area
        st.subheader("Sentiment Share Over Time (Stacked)")
        monthly_pct = monthly.copy()
        totals = monthly_pct.groupby("month")["count"].transform("sum")
        monthly_pct["pct"] = monthly_pct["count"] / totals * 100

        fig_area = px.area(
            monthly_pct, x="month", y="pct",
            color="final_label",
            color_discrete_map=COLORS,
            labels={"month": "Month", "pct": "% of Tweets", "final_label": "Sentiment"}
        )
        fig_area.update_layout(xaxis_tickangle=-45, hovermode="x unified")
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No timestamp data available for trend analysis.")


# ══════════════════════════════════════════
#  TAB 3 — EXPLAINABILITY
# ══════════════════════════════════════════
with tab3:
    st.subheader("🔍 What Words Drive Each Sentiment?")
    st.markdown("Using **SHAP values** to show which words most influenced the model's predictions.")

    if shap_df is not None:
        selected_class = st.radio(
            "Select sentiment class:",
            ["negative", "neutral", "positive"],
            horizontal=True
        )

        class_df = shap_df[shap_df["class"] == selected_class].head(15)

        fig_shap = px.bar(
            class_df.sort_values("importance"),
            x="importance", y="word",
            orientation="h",
            color_discrete_sequence=[COLORS[selected_class]],
            labels={"importance": "Mean |SHAP value|", "word": "Word"}
        )
        fig_shap.update_layout(
            title=f"Top words driving → {selected_class.upper()} predictions",
            margin=dict(t=40, b=20),
            height=450
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # side by side comparison
        st.subheader("All Classes Side by Side")
        fig_all = make_subplots(rows=1, cols=3,
                                subplot_titles=["NEGATIVE", "NEUTRAL", "POSITIVE"])

        for i, cls in enumerate(["negative", "neutral", "positive"], 1):
            cls_data = shap_df[shap_df["class"] == cls].head(10)
            fig_all.add_trace(
                go.Bar(
                    x=cls_data["importance"],
                    y=cls_data["word"],
                    orientation="h",
                    marker_color=COLORS[cls],
                    name=cls
                ),
                row=1, col=i
            )

        fig_all.update_layout(height=400, showlegend=False,
                               margin=dict(t=40, b=20))
        st.plotly_chart(fig_all, use_container_width=True)

    else:
        st.warning("SHAP data not found. Run `python src/explain.py` first.")


# ══════════════════════════════════════════
#  TAB 4 — LIVE PREDICTOR
# ══════════════════════════════════════════
with tab4:
    st.subheader("🤖 Live Sentiment Predictor")
    st.markdown("Type any tweet about Rwandan politics and the model will classify it.")

    user_input = st.text_area(
        "Enter a tweet:",
        placeholder="e.g. Rwanda's economy continues to grow under strong leadership...",
        height=100
    )

    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec     = tfidf.transform([cleaned])
            pred    = lr.predict(vec)[0]
            proba   = lr.predict_proba(vec)[0]
            classes = lr.classes_

            # result
            emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}
            color = COLORS[pred]

            st.markdown(f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        padding:1rem; border-radius:8px; margin:1rem 0;">
                <h3 style="color:{color}; margin:0">
                    {emoji.get(pred, '')} Prediction: {pred.upper()}
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # confidence bars
            st.markdown("**Confidence Scores:**")
            for cls, prob in sorted(zip(classes, proba),
                                    key=lambda x: x[1], reverse=True):
                st.markdown(f"{emoji.get(cls,'')} **{cls}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")

            # cleaned text
            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned)
        else:
            st.warning("Please enter some text first.")