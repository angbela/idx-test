import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="IDX Stochastic Buy Signal Scanner",
    layout="wide"
)

st.title("📈 IDX Stochastic Buy Signal Scanner")

# =========================
# Initialize session state
# =========================
if "scanned" not in st.session_state:
    st.session_state.scanned = False
    st.session_state.result_df = None
    st.session_state.data_cache = {}

# =========================
# User Inputs
# =========================
tickers_input = st.text_area(
    "Enter IDX tickers (one per line, without .JK)",
    placeholder="BBCA\nBMRI\nTLKM"
)

length = st.sidebar.number_input("Stochastic Length", 5, 30, 10)
smoothK = st.sidebar.number_input("Smooth K", 1, 10, 5)
smoothD = st.sidebar.number_input("Smooth D", 1, 10, 5)

run_btn = st.button("Run Scan")

# =========================
# Indicator functions
# =========================
def slow_stochastic(df, length, smoothK, smoothD):
    low_min = df["Low"].rolling(length).min()
    high_max = df["High"].rolling(length).max()

    fast_k = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["%K"] = fast_k.rolling(smoothK).mean()
    df["%D"] = df["%K"].rolling(smoothD).mean()
    return df

def find_last_buy_signal(df):
    condition = (
        (df["%K"].shift(1) < df["%D"].shift(1)) &
        (df["%K"] > df["%D"]) &
        (df["%K"] < 20) &
        (df["%D"] < 20)
    )
    return df[condition].index[-1] if condition.any() else None

# =========================
# Run scan (ONLY ONCE)
# =========================
if run_btn and tickers_input.strip():

    tickers = [
        t.strip().upper() + ".JK"
        for t in tickers_input.splitlines()
        if t.strip()
    ]

    total = len(tickers)
    st.subheader(f"Scanning {total} tickers")

    progress_bar = st.progress(0)
    status_text = st.empty()

    rows = []
    data_cache = {}

    for i, ticker in enumerate(tickers, start=1):
        status_text.text(f"Scanning {i}/{total} : {ticker}")
        progress_bar.progress(i / total)

        try:
            df = yf.Ticker(ticker).history(period="1y", interval="1d")
            if df.empty:
                continue

            # % change (for hover)
            df["PctChange"] = df["Close"].pct_change() * 100

            df = slow_stochastic(df, length, smoothK, smoothD)
            data_cache[ticker] = df.copy()

            last_volume = int(df["Volume"].iloc[-1])
            last_buy = find_last_buy_signal(df)

            rows.append({
                "Ticker": ticker.replace(".JK", ""),
                "LastBuySignal": last_buy.date() if last_buy else None,
                "Volume": last_volume
            })

        except Exception:
            rows.append({
                "Ticker": ticker.replace(".JK", ""),
                "LastBuySignal": None,
                "Volume": None
            })

    status_text.text("✅ Scan completed")
    progress_bar.progress(1.0)

    st.session_state.result_df = pd.DataFrame(rows)
    st.session_state.data_cache = data_cache
    st.session_state.scanned = True

# =========================
# Display results (NO RESET)
# =========================
if st.session_state.scanned:

    df = st.session_state.result_df

    show_only_buy = st.checkbox("Show only stocks with buy signal", value=True)
    if show_only_buy:
        df = df[df["LastBuySignal"].notna()]

    df = df.sort_values("LastBuySignal", ascending=False)

    st.subheader("📊 Scan Results")
    st.dataframe(df, use_container_width=True)

    # =========================
    # Chart selection
    # =========================
    if not df.empty:
        selected_ticker = st.selectbox(
            "Select ticker to view chart",
            df["Ticker"].tolist()
        )

        full_ticker = selected_ticker + ".JK"
        df_chart = st.session_state.data_cache.get(full_ticker)

        if df_chart is not None:
            st.subheader(f"📉 {selected_ticker} Candlestick, Volume & Stochastic")

            # Create subplot: price+stochastic / volume
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price & Stochastic", "Volume")
            )

            # Candlestick with % change hover
            fig.add_trace(
                go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["Open"],
                    high=df_chart["High"],
                    low=df_chart["Low"],
                    close=df_chart["Close"],
                    customdata=df_chart["PctChange"],
                    hovertemplate=(
                        "Date: %{x}<br>"
                        "Open: %{open}<br>"
                        "High: %{high}<br>"
                        "Low: %{low}<br>"
                        "Close: %{close} (%{customdata:+.2f}%)<extra></extra>"
                    ),
                    name="Price"
                ),
                row=1, col=1
            )

            # Stochastic lines
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=df_chart["%K"],
                    mode="lines",
                    name="%K"
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=df_chart["%D"],
                    mode="lines",
                    name="%D"
                ),
                row=1, col=1
            )

            fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)

            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=df_chart.index,
                    y=df_chart["Volume"],
                    name="Volume"
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
