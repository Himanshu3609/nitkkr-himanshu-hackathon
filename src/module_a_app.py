import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
import plotly.express as px
import re
import requests
from bs4 import BeautifulSoup
import feedparser

# Add src to path to import engine and data_ingestion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_engine_pipeline import RiskEnginePipeline
from risk_data_ingestion import download_datasets, DataStreamer

# Initialize Streamlit Page
st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")

# Prevent screen dimming / brightness pulsing animation on rerun
st.markdown("""
<style>
/* Disable opacity fading on Streamlit reruns */
[data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], .stApp {
    transition: none !important;
}
div.element-container, div.stMarkdown, div.stDataFrame, div[data-testid="stMetric"], .stPlotlyChart {
    opacity: 1 !important;
    transition: none !important;
}
[data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("AI/NLP Risk Engine & Tactical Index Rebalancing")
st.markdown("""
This dashboard demonstrates **Module A**. 
1. The **Core AI Pipeline** ingests structured/unstructured text and computes Sentiment, Event Classification, and Impact.
2. The **Tactical Index Rebalancer** adjusts a mock portfolio dynamically based on real-time sentiment signals.
""")

# Initialize Mock Portfolio in Session State
INITIAL_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

if 'tickers' not in st.session_state:
    st.session_state.tickers = INITIAL_TICKERS.copy()

if 'company_multiselect' not in st.session_state:
    st.session_state.company_multiselect = INITIAL_TICKERS.copy()

if 'initial_allocations' not in st.session_state:
    st.session_state.initial_allocations = {ticker: 100.0 / len(INITIAL_TICKERS) for ticker in INITIAL_TICKERS}

if 'portfolio' not in st.session_state:
    # Equal weight initially based on the number of initial tickers
    weight = 100.0 / len(st.session_state.tickers)
    st.session_state.portfolio = {ticker: weight for ticker in st.session_state.tickers}
    st.session_state.history = pd.DataFrame(columns=['Iteration'] + st.session_state.tickers)
    
    initial_row = {'Iteration': 0}
    initial_row.update(st.session_state.portfolio)
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([initial_row])], ignore_index=True)
    
    st.session_state.iteration = 0

# Table to organize live feeds
if 'feed_table' not in st.session_state:
    st.session_state.feed_table = pd.DataFrame(columns=["Timestamp", "Source", "Headline", "Ticker", "Sentiment", "Event", "Impact"])

@st.cache_resource(show_spinner="Loading optimized NLP Models (FinBERT & DistilBERT MNLI) into memory...")
def load_engine():
    return RiskEnginePipeline()

if 'engine' not in st.session_state:
    st.session_state.engine = load_engine()
    st.success("Models Loaded!")

st.header("1. Live Feed & Risk Analysis Pipeline")

input_method = st.radio("Select Data Source", [
    "Live News RSS Feeds (Crawler)", 
    "Process Tweet by URL", 
    "Live Stream (Kaggle Datasets)", 
    "Manual Text Input"
], horizontal=True)

def fetch_tweet_text(url_or_id):
    """
    Fetches real tweet text dynamically by ID or URL using a public embed API.
    """
    tweet_id = re.search(r'status/(\d+)', url_or_id)
    if tweet_id:
        tweet_id = tweet_id.group(1)
    else:
        tweet_id = re.search(r'(\d+)', url_or_id)
        if tweet_id:
            tweet_id = tweet_id.group(1)
        else:
            return "Error: Could not extract a valid Tweet ID from the input."
            
    try:
        res = requests.get(f"https://api.fxtwitter.com/status/{tweet_id}", timeout=10)
        data = res.json()
        
        if data.get('code') == 200 and 'tweet' in data:
            return data['tweet']['text']
        else:
            return f"Error: {data.get('message', 'Tweet not found or deleted.')}"
    except Exception as e:
        return f"Error fetching tweet: {str(e)}"

def get_affected_ticker(text):
    """Smart extraction of tickers, dynamically discovering new ones from news and tweets."""
    text_lower = text.lower()
    
    cashtags = re.findall(r'\$([a-zA-Z]{1,5})\b', text)
    if cashtags:
        return cashtags[0].upper()
        
    company_map = {
        'apple': 'AAPL', 'iphone': 'AAPL', 'mac': 'AAPL',
        'microsoft': 'MSFT', 'windows': 'MSFT', 'azure': 'MSFT',
        'google': 'GOOGL', 'alphabet': 'GOOGL', 'youtube': 'GOOGL',
        'amazon': 'AMZN', 'aws': 'AMZN', 'prime': 'AMZN',
        'tesla': 'TSLA', 'musk': 'TSLA', 'model 3': 'TSLA',
        'nvidia': 'NVDA', 'jensen': 'NVDA', 'geforce': 'NVDA',
        'meta': 'META', 'facebook': 'META', 'zuckerberg': 'META',
        'netflix': 'NFLX', 'salesforce': 'CRM', 'disney': 'DIS',
        'intel': 'INTC', 'amd': 'AMD', 'palantir': 'PLTR',
        'coinbase': 'COIN', 'bitcoin': 'BTC', 'ethereum': 'ETH',
        'jpmorgan': 'JPM', 'goldman': 'GS', 'bank of america': 'BAC',
        'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST',
        'boeing': 'BA', 'pfizer': 'PFE', 'johnson & johnson': 'JNJ',
        'reliance': 'RELIANCE', 'tata': 'TCS', 'infosys': 'INFY'
    }
    
    for kw, mapped_ticker in company_map.items():
        if re.search(rf'\b{kw}\b', text_lower):
            return mapped_ticker
            
    return 'SPY' # Broad market proxy

def update_portfolio(ticker, sentiment, impact):
    """Core rebalancing logic based on signal"""
    if ticker not in st.session_state.tickers:
        st.session_state.tickers.append(ticker)
        if 'company_multiselect' in st.session_state and ticker not in st.session_state.company_multiselect:
            st.session_state.company_multiselect.append(ticker)
            
        st.session_state.portfolio[ticker] = 2.0
        if 'initial_allocations' in st.session_state and ticker not in st.session_state.initial_allocations:
            st.session_state.initial_allocations[ticker] = 2.0
            
        for t in st.session_state.portfolio:
            if t != ticker:
                st.session_state.portfolio[t] -= (2.0 / (len(st.session_state.tickers)-1))
        
        st.session_state.history[ticker] = 0.0
        st.success(f"🌟 New Asset Discovered! Automatically added **{ticker}** to the portfolio tracking.")

    current_weight = st.session_state.portfolio[ticker]
    adjustment = sentiment * (impact / 10.0) * 10.0 
    new_weight = max(2.0, current_weight + adjustment)
    st.session_state.portfolio[ticker] = new_weight
    
    total = sum(st.session_state.portfolio.values())
    st.session_state.portfolio = {k: (v/total)*100 for k, v in st.session_state.portfolio.items()}
    
    st.session_state.iteration += 1
    new_row = {'Iteration': st.session_state.iteration}
    new_row.update(st.session_state.portfolio)
    
    for col in st.session_state.history.columns:
        if col not in new_row:
            new_row[col] = 0.0
            
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

def append_to_feed_table(source, text, ticker, sentiment, event, impact):
    new_entry = pd.DataFrame([{
        "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Source": source,
        "Headline": text,
        "Ticker": ticker,
        "Sentiment": round(sentiment, 2),
        "Event": event,
        "Impact": impact
    }])
    st.session_state.feed_table = pd.concat([new_entry, st.session_state.feed_table], ignore_index=True)


should_auto_rerun = False

if input_method == "Live News RSS Feeds (Crawler)":
    st.info("📡 Web Crawler ingesting live financial news from Google News and MoneyControl RSS feeds.")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        feed_choice = st.selectbox("Select Feed Source to Crawl", ["Google News (Finance)", "MoneyControl (Business)", "MoneyControl (Markets)"])
    with col2:
        auto_crawl = st.toggle("🚀 Run Continuous Auto-Crawler")
        
    feed_urls = {
        "Google News (Finance)": "https://news.google.com/rss/search?q=finance+OR+stock+market&hl=en-US&gl=US&ceid=US:en",
        "MoneyControl (Business)": "https://www.moneycontrol.com/rss/business.xml",
        "MoneyControl (Markets)": "https://www.moneycontrol.com/rss/marketreports.xml"
    }
    
    # Initialize queue and counts
    if 'rss_queue' not in st.session_state:
        st.session_state.rss_queue = []
    if 'processed_rss_count' not in st.session_state:
        st.session_state.processed_rss_count = 0

    metric_col, manual_col = st.columns([1, 2])
    with metric_col:
        st.metric("Total Articles Processed", st.session_state.processed_rss_count)
        
    with manual_col:
        # User input for 'cnt' instead of a slider
        cnt_val = st.number_input("Enter count of recent articles to process (cnt):", min_value=1, max_value=50, value=3, step=1)
        if st.button(f"Fetch & Process Recent {cnt_val} Articles"):
            with st.spinner(f"Crawling latest {cnt_val} articles from {feed_choice}..."):
                feed = feedparser.parse(feed_urls[feed_choice])
                entries = feed.entries[:cnt_val]
                
            if not entries:
                st.error("Feed returned no items or could not be reached.")
            else:
                for entry in entries:
                    text = entry.title
                    st.write(f"📰 **[{feed_choice}]** {text}")
                    result = st.session_state.engine.run_pipeline(text, source=feed_choice)
                    affected = get_affected_ticker(text)
                    
                    update_portfolio(affected, result['sentiment_score'], result['impact_score'])
                    append_to_feed_table(feed_choice, text, affected, result['sentiment_score'], result['event_class'], result['impact_score'])
                    st.session_state.processed_rss_count += 1
                st.success(f"Successfully processed {len(entries)} articles!")
    
    # Dynamic continuous background crawler
    if auto_crawl:
        live_status = st.empty()
        
        # Populate queue if empty
        if not st.session_state.rss_queue:
            feed = feedparser.parse(feed_urls[feed_choice])
            st.session_state.rss_queue = [entry.title for entry in feed.entries[:25]]
            
        if st.session_state.rss_queue:
            text = st.session_state.rss_queue.pop(0)
            live_status.info(f"⚡ [Dynamic Crawler Processing]: {text}")
            
            result = st.session_state.engine.run_pipeline(text, source=feed_choice)
            affected = get_affected_ticker(text)
            
            update_portfolio(affected, result['sentiment_score'], result['impact_score'])
            append_to_feed_table(feed_choice, text, affected, result['sentiment_score'], result['event_class'], result['impact_score'])
            
            st.session_state.processed_rss_count += 1
            should_auto_rerun = True
        else:
            live_status.warning("Waiting for new live headlines...")
            should_auto_rerun = True

elif input_method == "Process Tweet by URL":
    tweet_url = st.text_input("Enter a Twitter / X post URL or ID:", "https://twitter.com/BarackObama/status/896523232098078720")
    if st.button("Fetch & Analyze"):
        with st.spinner("Fetching real tweet..."):
            fetched_text = fetch_tweet_text(tweet_url)
            
        if fetched_text.startswith("Error:"):
            st.error(fetched_text)
        else:
            st.info(f"**Fetched Text:** {fetched_text}")
            with st.spinner("Running Pipeline..."):
                result = st.session_state.engine.run_pipeline(fetched_text, source="Twitter")
                affected_ticker = get_affected_ticker(fetched_text)
                
                update_portfolio(affected_ticker, result['sentiment_score'], result['impact_score'])
                append_to_feed_table("Twitter URL", fetched_text, affected_ticker, result['sentiment_score'], result['event_class'], result['impact_score'])
                
                # Instant visual feedback
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
                col2.metric("Event Classification", result['event_class'])
                col3.metric("Impact Score (1-10)", result['impact_score'])
                col4.metric("Mapped Ticker", affected_ticker)
                st.success(f"✅ Processed tweet and dynamically rebalanced **{affected_ticker}**! Signal added to Feed Table.")

elif input_method == "Live Stream (Kaggle Datasets)":
    if 'streamer' not in st.session_state:
        st.session_state.streamer = None
        
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Initialize Kaggle Datasets"):
            with st.spinner("Downloading and parsing News & Tweets..."):
                n_path, t_path = download_datasets()
                st.session_state.streamer = DataStreamer(n_path, t_path)
            st.success("Ready!")
            
    with col_b:
        if st.session_state.streamer:
            col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
            with col_b1:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=5)
            with col_b2:
                process_batch_btn = st.button("Consume Batch")
            with col_b3:
                process_full_btn = st.button("Process Full Remaining Dataset")
                
            items_to_process = 0
            if process_batch_btn:
                items_to_process = batch_size
            elif process_full_btn:
                items_to_process = 1000000 
                
            if items_to_process > 0:
                with st.spinner("Processing feed..."):
                    processed_count = 0
                    for _ in range(items_to_process):
                        item = st.session_state.streamer.get_next()
                        if item:
                            icon = "🐦" if item['source'] == "Twitter" else "📰"
                            result = st.session_state.engine.run_pipeline(item['text'], source=item['source'])
                            affected = get_affected_ticker(item['text'])
                            
                            update_portfolio(affected, result['sentiment_score'], result['impact_score'])
                            append_to_feed_table(item['source'], item['text'], affected, result['sentiment_score'], result['event_class'], result['impact_score'])
                            
                            st.write(f"**{icon} [{item['source']}]** {item['text'][:120]}... ➔ *{affected}* (Sent: `{result['sentiment_score']:.2f}` | Event: `{result['event_class']}`)")
                            processed_count += 1
                        else:
                            st.warning("End of stream reached.")
                            break
                    if processed_count > 0:
                        st.success(f"✅ Processed {processed_count} items! Portfolio rebalanced and Feed Table updated.")

elif input_method == "Manual Text Input":
    user_text = st.text_area("Enter a financial news headline or tweet:", "Nvidia hits record high on AI chip demand. $NVDA soaring.")
    if st.button("Process through Pipeline"):
        with st.spinner("Running Pipeline..."):
            result = st.session_state.engine.run_pipeline(user_text, source="Manual")
            affected_ticker = get_affected_ticker(user_text)
            
            update_portfolio(affected_ticker, result['sentiment_score'], result['impact_score'])
            append_to_feed_table("Manual Input", user_text, affected_ticker, result['sentiment_score'], result['event_class'], result['impact_score'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
            col2.metric("Event Classification", result['event_class'])
            col3.metric("Impact Score (1-10)", result['impact_score'])
            col4.metric("Mapped Ticker", affected_ticker)
            st.success(f"✅ Processed headline and rebalanced **{affected_ticker}**! Logged to Feed Table.")


st.divider()

# ==============================================================================
# SECTION 2: ORGANIZED FEED TABLE
# ==============================================================================
st.header("2. Mock Stock Index & Live Feed Audit Table")

st.subheader(f"Recent Processed Signals ({len(st.session_state.feed_table)} Total Records)")
if not st.session_state.feed_table.empty:
    st.dataframe(
        st.session_state.feed_table, 
        use_container_width=True, 
        hide_index=True
    )
else:
    st.info("ℹ️ No signals processed yet. Run the crawler, consume a batch, or enter a tweet above to populate live records.")

st.divider()

# ==============================================================================
# SECTION 3: SENTIMENT FACTOR ATTRIBUTION & BREAKDOWN
# ==============================================================================
st.header("3. Sentiment Factor Attribution & Macro Risk Breakdown")
st.markdown("Quantifying the **percentage impact of different corporate, geopolitical, and macroeconomic factors** actively moving market sentiments.")

# Calculate factor breakdown
if not st.session_state.feed_table.empty:
    factor_df = st.session_state.feed_table['Event'].value_counts().reset_index()
    factor_df.columns = ['Factor', 'Count']
    factor_df['Percentage (%)'] = ((factor_df['Count'] / factor_df['Count'].sum()) * 100).round(1)
    
    # Calculate avg sentiment per factor
    avg_sent = st.session_state.feed_table.groupby('Event')['Sentiment'].mean().reset_index()
    avg_sent.columns = ['Factor', 'Avg Sentiment']
    factor_df = factor_df.merge(avg_sent, on='Factor')
    factor_df['Avg Sentiment'] = factor_df['Avg Sentiment'].round(3)
else:
    # Baseline benchmark distribution if no signals processed yet
    default_factors = [
        {"Factor": "Macroeconomic Shift", "Percentage (%)": 28.5, "Avg Sentiment": 0.12, "Count": 14},
        {"Factor": "Earnings Report", "Percentage (%)": 22.0, "Avg Sentiment": 0.45, "Count": 11},
        {"Factor": "Geopolitical Event", "Percentage (%)": 16.0, "Avg Sentiment": -0.58, "Count": 8},
        {"Factor": "Regulatory Action", "Percentage (%)": 14.0, "Avg Sentiment": -0.35, "Count": 7},
        {"Factor": "Product Launch", "Percentage (%)": 11.5, "Avg Sentiment": 0.62, "Count": 6},
        {"Factor": "Credit Risk", "Percentage (%)": 8.0, "Avg Sentiment": -0.71, "Count": 4}
    ]
    factor_df = pd.DataFrame(default_factors)

col_f1, col_f2 = st.columns([1.2, 1.8])

with col_f1:
    fig_factor = px.pie(
        factor_df, 
        values='Percentage (%)', 
        names='Factor', 
        hole=0.55,
        title="Distribution of Sentiment Drivers (%)",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_factor.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="<b>%{label}</b><br>Share: %{percent}<br>Count: %{value}")
    fig_factor.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_factor, use_container_width=True)

with col_f2:
    fig_sent_bar = px.bar(
        factor_df, 
        x='Avg Sentiment', 
        y='Factor', 
        orientation='h',
        color='Avg Sentiment',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="Average Sentiment Bias by Factor (-1.0 Bearish to +1.0 Bullish)",
        text_auto='.2f'
    )
    fig_sent_bar.update_layout(
        margin=dict(t=40, b=10, l=10, r=10), 
        coloraxis_showscale=False,
        xaxis_range=[-1.0, 1.0],
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_sent_bar, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: COMPANY-WISE GROWTH, FORECASTS & FALLOUT PATTERNS
# ==============================================================================
st.header("4. Company-Wise Allocation, Growth & Fallout Pattern Analysis")
st.markdown("Detailed breakdown tracking **Initial Allocation**, **Current Weight**, **Net Growth**, **Recent Trajectory Pattern (Growth vs. Fallout)**, and **AI Future Predictions**.")

# Compute company performance metrics
perf_rows = []

for ticker in st.session_state.tickers:
    init_wt = st.session_state.initial_allocations.get(ticker, 2.0)
    curr_wt = st.session_state.portfolio.get(ticker, 0.0)
    net_growth = curr_wt - init_wt
    pct_growth = (net_growth / init_wt) * 100 if init_wt > 0 else 0.0
    
    # Trajectory pattern (Growth vs Fallout)
    hist_series = st.session_state.history[ticker] if ticker in st.session_state.history else pd.Series([curr_wt])
    if len(hist_series) >= 2:
        recent_delta = hist_series.iloc[-1] - hist_series.iloc[-2]
    else:
        recent_delta = 0.0
        
    if recent_delta > 1.2:
        pattern = "🚀 Strong Momentum (Surging)"
        pattern_type = "Growth"
    elif recent_delta > 0.1:
        pattern = "📈 Steady Growth"
        pattern_type = "Growth"
    elif recent_delta < -1.2:
        pattern = "🔻 Severe Fallout (De-risking)"
        pattern_type = "Fallout"
    elif recent_delta < -0.1:
        pattern = "📉 Selling Pressure / Pullback"
        pattern_type = "Fallout"
    else:
        pattern = "⚖️ Neutral / Stable"
        pattern_type = "Stable"
        
    # Future AI Prediction (Momentum + Mean-reversion projection)
    if len(hist_series) >= 3:
        momentum_slope = (hist_series.iloc[-1] - hist_series.iloc[-3]) / 2.0
    else:
        momentum_slope = recent_delta
        
    predicted_raw = max(2.0, curr_wt + (momentum_slope * 1.2))
    
    # Find dominant sentiment factor for this ticker from feed_table
    if not st.session_state.feed_table.empty and (st.session_state.feed_table['Ticker'] == ticker).any():
        ticker_signals = st.session_state.feed_table[st.session_state.feed_table['Ticker'] == ticker]
        dom_factor = ticker_signals['Event'].mode()[0] if not ticker_signals.empty else "Market Flow"
    else:
        dom_factor = "Broad Sentiment"

    perf_rows.append({
        "Ticker": ticker,
        "Initial Allocation (%)": round(init_wt, 2),
        "Current Allocation (%)": round(curr_wt, 2),
        "Net Growth (%)": round(net_growth, 2),
        "Relative Return (%)": round(pct_growth, 1),
        "Recent Trajectory": pattern,
        "Pattern Type": pattern_type,
        "raw_pred": predicted_raw,
        "Dominant Factor": dom_factor
    })

perf_df = pd.DataFrame(perf_rows)

# Normalize predicted weights so total is 100%
pred_sum = perf_df['raw_pred'].sum()
perf_df['Future Prediction (%)'] = ((perf_df['raw_pred'] / pred_sum) * 100).round(2)
perf_df.drop(columns=['raw_pred'], inplace=True)

# KPI Highlight Cards
top_growth_asset = perf_df.loc[perf_df['Net Growth (%)'].idxmax()]
top_fallout_asset = perf_df.loc[perf_df['Net Growth (%)'].idxmin()]
top_predicted_asset = perf_df.loc[perf_df['Future Prediction (%)'].idxmax()]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        "🏆 Top Growth Asset", 
        f"{top_growth_asset['Ticker']}", 
        f"+{top_growth_asset['Net Growth (%)']:.1f}% ({top_growth_asset['Current Allocation (%)']}%)"
    )
with kpi2:
    fallout_val = top_fallout_asset['Net Growth (%)']
    st.metric(
        "⚠️ Greatest Fallout / Drawdown", 
        f"{top_fallout_asset['Ticker']}", 
        f"{fallout_val:.1f}% ({top_fallout_asset['Current Allocation (%)']}%)",
        delta_color="inverse"
    )
with kpi3:
    st.metric(
        "🔮 Highest Projected Asset (Next Cycle)", 
        f"{top_predicted_asset['Ticker']}", 
        f"Target: {top_predicted_asset['Future Prediction (%)']}%"
    )
with kpi4:
    dominant_market_event = factor_df.iloc[0]['Factor'] if not factor_df.empty else "Macroeconomic Shift"
    st.metric(
        "🧭 Dominant Market Risk Driver", 
        dominant_market_event, 
        f"{factor_df.iloc[0]['Percentage (%)']}% Share" if not factor_df.empty else "35% Share"
    )

# Display the styled Performance Table
st.subheader("Company Performance, Fallout & AI Forecast Matrix")
display_perf = perf_df[['Ticker', 'Initial Allocation (%)', 'Current Allocation (%)', 'Net Growth (%)', 'Relative Return (%)', 'Recent Trajectory', 'Future Prediction (%)', 'Dominant Factor']]
st.dataframe(
    display_perf, 
    use_container_width=True, 
    hide_index=True
)

st.divider()

# ==============================================================================
# SECTION 5: ADVANCED VISUALIZATIONS & INTERACTIVE CHARTS
# ==============================================================================
st.header("5. Tactical Rebalancing Visualizations & Asset Dynamics")

tab_spline, tab_bar, tab_donut = st.tabs([
    "📈 Historical Trajectory Curves",
    "📊 3-Way Allocation Shift (Initial vs Current vs Prediction)",
    "🍩 Current Portfolio Donut"
])

with tab_spline:
    st.subheader("Interactive Historical Trend by Company")
    if not st.session_state.history.empty:
        hist_df = st.session_state.history.copy()
        all_available_tickers = [t for t in st.session_state.tickers if t in hist_df.columns]
        
        c_sel1, c_sel2 = st.columns([2.5, 1.5])
        with c_sel1:
            selected_tickers = st.multiselect(
                "Select Companies to Display on Historical Curve:",
                options=all_available_tickers,
                default=all_available_tickers,
                help="Choose which companies you want plotted on the chart."
            )
        with c_sel2:
            focused_company = st.selectbox(
                "Highlight Focused Company Over Curves:",
                options=["None (Equal Display)"] + (selected_tickers if selected_tickers else all_available_tickers),
                help="Make a specific company line thicker with direct data labels on the curve."
            )
            
        if selected_tickers:
            melted_df = hist_df.melt(id_vars=['Iteration'], value_vars=selected_tickers, var_name='Ticker', value_name='Weight (%)')
            
            fig_line = px.line(
                melted_df, 
                x='Iteration', 
                y='Weight (%)', 
                color='Ticker', 
                markers=True,
                title="Tactical Portfolio Weight Rebalancing Trajectory",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            
            # Format traces based on focused company selection
            for trace in fig_line.data:
                if focused_company != "None (Equal Display)":
                    if trace.name == focused_company:
                        trace.line.width = 4.5
                        trace.marker.size = 8
                        trace.mode = 'lines+markers+text'
                        trace.text = [f"{v:.1f}%" for v in trace.y]
                        trace.textposition = "top center"
                    else:
                        trace.line.width = 1.8
                        trace.opacity = 0.4
                        trace.marker.size = 4
                else:
                    trace.line.width = 2.5
                    trace.marker.size = 6

            fig_line.update_traces(line=dict(shape='linear'))
            fig_line.update_layout(
                yaxis_range=[0, 100], 
                margin=dict(t=40, b=10, l=10, r=10), 
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("Please select at least one company to view the graph.")

with tab_bar:
    st.subheader("Allocation Shift: Baseline Initial vs. Current Weight vs. Future AI Prediction")
    bar_df = perf_df[['Ticker', 'Initial Allocation (%)', 'Current Allocation (%)', 'Future Prediction (%)']].melt(
        id_vars=['Ticker'], 
        var_name='Stage', 
        value_name='Allocation (%)'
    )
    fig_comp = px.bar(
        bar_df, 
        x='Ticker', 
        y='Allocation (%)', 
        color='Stage', 
        barmode='group',
        title="3-Way Portfolio Rebalancing Comparison per Company",
        color_discrete_map={
            'Initial Allocation (%)': '#4A90E2',
            'Current Allocation (%)': '#F5A623',
            'Future Prediction (%)': '#2ECC71'
        },
        text_auto='.1f'
    )
    fig_comp.update_layout(
        yaxis_range=[0, max(50.0, perf_df['Current Allocation (%)'].max() + 10)],
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab_donut:
    st.subheader("Current Index Allocations (Donut Composition)")
    current_df = pd.DataFrame(list(st.session_state.portfolio.items()), columns=['Ticker', 'Weight (%)'])
    current_df['Weight (%)'] = current_df['Weight (%)'].round(2)
    
    fig_pie = px.pie(
        current_df, 
        values='Weight (%)', 
        names='Ticker', 
        hole=0.55,
        title="Live Index Weight Share (%)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

# Auto-rerun trigger at the very bottom of the page AFTER all widgets, tables, and graphs are drawn
if should_auto_rerun:
    time.sleep(2.0)
    st.rerun()
