import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
# API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="StockPredo — Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 StockPredo")
    st.caption("Sentiment Analysis Engine · Module 1")
    st.divider()

    ticker_input = st.text_input(
        "Enter ticker symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT",
    ).upper().strip()

    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    st.divider()

    compare_input = st.text_input(
        "Compare multiple tickers",
        placeholder="AAPL,MSFT,TSLA",
    )
    compare_btn = st.button("📊 Compare", use_container_width=True)

    st.divider()
    st.caption("Data sources: yfinance · NewsAPI · Reddit · RSS")
    st.caption("Models: FinBERT (70%) + VADER (30%)")


# ─── Helper functions ────────────────────────────────────────────────────────
def signal_color(signal: str) -> str:
    return {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(signal, "⚪")

def score_to_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 3),
        delta={"reference": 0},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [-1, -0.05], "color": "#ff4b4b"},
                {"range": [-0.05, 0.05], "color": "#ffd700"},
                {"range": [0.05, 1], "color": "#2ecc71"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": "Ensemble Sentiment Score"},
        number={"font": {"size": 40}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig

def fetch_summary(ticker: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/sentiment/{ticker}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_articles(ticker: str) -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE}/articles/{ticker}?limit=100", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def run_analysis(ticker: str):
    with st.spinner(f"Scraping and analyzing {ticker}... (FinBERT first run downloads ~400MB)"):
        try:
            r = httpx.post(f"{API_BASE}/analyze/{ticker}", timeout=300)
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"API error: {r.text}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
    return None


# ─── Main panel ──────────────────────────────────────────────────────────────
st.title("StockPredo — Real-time Market Sentiment")

if analyze_btn and ticker_input:
    result = run_analysis(ticker_input)
    if result:
        st.success(f"Analysis complete for **{ticker_input}**")

# Load existing data for the current ticker
summary = fetch_summary(ticker_input) if ticker_input else None
articles = fetch_articles(ticker_input) if ticker_input else []

if summary:
    sig = summary["signal"]
    emoji = signal_color(sig)

    # ─── Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal", f"{emoji} {sig}")
    col2.metric("Avg Sentiment Score", f"{summary['avg_score']:.4f}")
    col3.metric("Total Articles", summary["total_articles"])
    col4.metric(
        "Bullish / Bearish",
        f"{summary['bullish_count']} / {summary['bearish_count']}",
    )

    st.divider()

    # ─── Gauge + Distribution
    gcol, dcol = st.columns([1, 1])
    with gcol:
        st.plotly_chart(score_to_gauge(summary["avg_score"]), use_container_width=True)

    if articles:
        df = pd.DataFrame(articles)
        df["ensemble_score"] = pd.to_numeric(df["ensemble_score"], errors="coerce")
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        with dcol:
            fig_hist = px.histogram(
                df.dropna(subset=["ensemble_score"]),
                x="ensemble_score",
                nbins=30,
                color_discrete_sequence=["#1f77b4"],
                title="Score distribution across articles",
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_hist.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ─── Source breakdown
    if articles:
        df["source_group"] = df["source"].apply(
            lambda x: "Reddit" if "reddit" in str(x).lower()
            else "NewsAPI" if "newsapi" in str(x).lower()
            else "yfinance" if "yfinance" in str(x).lower()
            else "RSS"
        )

        src_agg = (
            df.groupby("source_group")["ensemble_score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_score", "count": "articles"})
        )

        col_src, col_pie = st.columns([1.5, 1])
        with col_src:
            fig_bar = px.bar(
                src_agg,
                x="source_group",
                y="avg_score",
                color="avg_score",
                color_continuous_scale=["#ff4b4b", "#ffd700", "#2ecc71"],
                range_color=[-1, 1],
                title="Average sentiment by source",
                labels={"source_group": "Source", "avg_score": "Avg score"},
            )
            fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_bar.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            pie_data = pd.DataFrame({
                "sentiment": ["Bullish", "Neutral", "Bearish"],
                "count": [summary["bullish_count"], summary["neutral_count"], summary["bearish_count"]],
            })
            fig_pie = px.pie(
                pie_data,
                names="sentiment",
                values="count",
                color="sentiment",
                color_discrete_map={"Bullish": "#2ecc71", "Neutral": "#ffd700", "Bearish": "#ff4b4b"},
                title="Sentiment breakdown",
            )
            fig_pie.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # ─── Timeline chart
        timeline_df = (
            df.dropna(subset=["published_at", "ensemble_score"])
            .sort_values("published_at")
        )
        if not timeline_df.empty:
            st.subheader("Sentiment over time")
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=timeline_df["published_at"],
                y=timeline_df["ensemble_score"],
                mode="markers",
                marker=dict(
                    color=timeline_df["ensemble_score"],
                    colorscale=[[0, "#ff4b4b"], [0.5, "#ffd700"], [1, "#2ecc71"]],
                    cmin=-1, cmax=1,
                    size=8,
                    showscale=True,
                ),
                text=timeline_df["headline"],
                hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<br>%{x}",
            ))
            fig_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_time.update_layout(
                height=350,
                margin=dict(t=20, b=0),
                yaxis_title="Sentiment score",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # ─── Articles table
        st.subheader("Articles with sentiment scores")
        display_df = df[["published_at", "source", "headline", "ensemble_score", "finbert_label", "url"]].copy()
        display_df = display_df.sort_values("published_at", ascending=False)
        display_df["ensemble_score"] = display_df["ensemble_score"].round(4)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "ensemble_score": st.column_config.NumberColumn(
                    "Score", format="%.4f", min_value=-1, max_value=1
                ),
            },
        )
elif ticker_input:
    st.info(f"No data for **{ticker_input}** yet. Click **Analyze Sentiment** to start.")

# ─── Compare panel ───────────────────────────────────────────────────────────
if compare_btn and compare_input:
    tickers_list = [t.strip().upper() for t in compare_input.split(",") if t.strip()]
    st.divider()
    st.subheader(f"Comparing: {', '.join(tickers_list)}")

    with st.spinner("Loading comparison..."):
        try:
            r = httpx.get(f"{API_BASE}/compare?tickers={compare_input}", timeout=15)
            if r.status_code == 200:
                cmp_data = r.json()
                if cmp_data:
                    cdf = pd.DataFrame(cmp_data)
                    fig_cmp = px.bar(
                        cdf,
                        x="ticker",
                        y="avg_score",
                        color="avg_score",
                        color_continuous_scale=["#ff4b4b", "#ffd700", "#2ecc71"],
                        range_color=[-1, 1],
                        title="Sentiment comparison",
                        text="avg_score",
                    )
                    fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_cmp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_cmp.update_layout(height=400)
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    st.dataframe(cdf, use_container_width=True)
                else:
                    st.warning("Analyze each ticker first before comparing.")
        except Exception as e:
            st.error(f"Compare failed: {e}")