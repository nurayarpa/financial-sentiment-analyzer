import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import feedparser
from transformers import pipeline
import torch

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# ── LOAD FINBERT (cached — loads only once) ───────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=0 if torch.cuda.is_available() else -1
    )

finbert = load_model()

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def get_news(ticker):
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    feed = feedparser.parse(url)
    
    rows = []
    for entry in feed.entries:
        text = entry.title + '. ' + entry.get('summary', '')
        result = finbert(text[:512])[0]
        score = result['score'] * (1 if result['label'] == 'positive' else -1)
        rows.append({
            'date': pd.to_datetime(entry.published).normalize().tz_localize(None),
            'title': entry.title,
            'sentiment': result['label'],
            'score': score
        })
    
    return pd.DataFrame(rows)

def get_prices(ticker, days):
    hist = yf.Ticker(ticker).history(period=f'{days}d')[['Close']].reset_index()
    hist.columns = ['date', 'price']
    hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None).dt.normalize()
    hist['price_change'] = hist['price'].pct_change() * 100
    return hist

# ── UI LAYOUT ─────────────────────────────────────────────────
st.title("📈 Financial News Sentiment Analyzer")
st.markdown("*Powered by FinBERT — trained on financial text*")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker  = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
company = st.sidebar.text_input("Company Name", value="Apple")
days    = st.sidebar.slider("Days of price history", 7, 30, 14)
analyze = st.sidebar.button("🔍 Analyze", type="primary")

# Instructions before first run
if not analyze:
    st.info("👈 Enter a stock ticker on the left and click **Analyze** to start")
    st.markdown("""
    ### How it works
    1. Fetches latest financial news from Yahoo Finance
    2. Runs each headline through **FinBERT** (AI model trained on financial text)
    3. Labels each article: 🟢 Positive / 🔴 Negative / ⚪ Neutral
    4. Overlays sentiment scores with real stock price data
    """)

# ── MAIN ANALYSIS ─────────────────────────────────────────────
if analyze:
    
    # Loading spinner
    with st.spinner(f"Fetching news and analyzing sentiment for {ticker}..."):
        news_df = get_news(ticker)
        prices  = get_prices(ticker, days)
    
    if news_df.empty:
        st.error(f"No news found for {ticker}. Try a different ticker.")
        st.stop()
    
    # Daily average sentiment
    daily = news_df.groupby('date')['score'].mean().reset_index()
    daily.columns = ['date', 'avg_sentiment']
    merged = pd.merge(prices, daily, on='date', how='left').fillna(0)
    
    # ── METRIC CARDS ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    
    current_price  = prices['price'].iloc[-1]
    price_change   = prices['price_change'].iloc[-1]
    pos_pct        = (news_df['sentiment'] == 'positive').mean() * 100
    neg_pct        = (news_df['sentiment'] == 'negative').mean() * 100
    corr_val       = merged[['price_change','avg_sentiment']].dropna().corr().iloc[0,1]
    
    col1.metric("Current Price",     f"${current_price:.2f}", f"{price_change:.2f}%")
    col2.metric("Articles Analyzed", len(news_df))
    col3.metric("Positive News",     f"{pos_pct:.0f}%")
    col4.metric("Sentiment-Price Corr", f"{corr_val:.3f}")
    
    st.markdown("---")
    
    # ── CHART ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Stock Price ($)', 'Avg News Sentiment'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08
    )
    
    # Price line
    fig.add_trace(go.Scatter(
        x=merged['date'], y=merged['price'],
        mode='lines', name='Price',
        line=dict(color='#2196F3', width=2)
    ), row=1, col=1)
    
    # Sentiment bars
    bar_colors = ['#4CAF50' if x > 0 else '#F44336' for x in merged['avg_sentiment']]
    fig.add_trace(go.Bar(
        x=merged['date'], y=merged['avg_sentiment'],
        name='Sentiment', marker_color=bar_colors
    ), row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title=f"{company} ({ticker}) — Price vs News Sentiment"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ── SENTIMENT BREAKDOWN ───────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("📊 Sentiment Breakdown")
        breakdown = news_df['sentiment'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=breakdown.index,
            values=breakdown.values,
            marker_colors=['#4CAF50', '#9E9E9E', '#F44336']
        ))
        fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_b:
        st.subheader("📰 Latest Headlines")
        for _, row in news_df.head(10).iterrows():
            icon = "🟢" if row['sentiment'] == 'positive' else (
                   "🔴" if row['sentiment'] == 'negative' else "⚪")
            st.write(f"{icon} **{row['sentiment'].upper()}** — {row['title']}")
    
    st.markdown("---")
    
    # ── CORRELATION NOTE ──────────────────────────────────────
    st.subheader("📌 Interpretation")
    st.info("""
    **About the correlation score:** A weak correlation is expected and normal.
    News sentiment is just ONE factor affecting stock price. Other factors include
    macro events, earnings reports, Fed decisions, and institutional flows.
    A stronger signal would require longer history, multiple news sources,
    and lag analysis (does today's news affect tomorrow's price?).
    """)

    