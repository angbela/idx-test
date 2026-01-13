import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="IDX Trading Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# Session state initialization
# ======================================================
if "scanner_scanned" not in st.session_state:
    st.session_state.scanner_scanned = False
    st.session_state.scanner_result_df = None
    st.session_state.scanner_data_cache = {}

if "optimizer_results" not in st.session_state:
    st.session_state.optimizer_results = None
    st.session_state.optimizer_best_df = None
    st.session_state.optimizer_best_strategy = None

# ======================================================
# SHARED FUNCTIONS - Technical Indicators
# ======================================================
def stochastic(df, length, smoothK, smoothD):
    """Calculate Stochastic Oscillator"""
    low_min = df["Low"].rolling(length).min()
    high_max = df["High"].rolling(length).max()
    fast_k = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["%K"] = fast_k.rolling(smoothK).mean()
    df["%D"] = df["%K"].rolling(smoothD).mean()
    return df

def rsi(df, period):
    """Calculate RSI"""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def macd(df, fast, slow, signal):
    """Calculate MACD"""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

def bollinger(df, length, std):
    """Calculate Bollinger Bands"""
    ma = df["Close"].rolling(length).mean()
    sd = df["Close"].rolling(length).std()
    df["BB_mid"] = ma
    df["BB_upper"] = ma + std * sd
    df["BB_lower"] = ma - std * sd
    return df

def moving_average(df, short_window, long_window):
    """Calculate Moving Averages"""
    df["MA_short"] = df["Close"].rolling(short_window).mean()
    df["MA_long"] = df["Close"].rolling(long_window).mean()
    return df

# ======================================================
# OPTIMIZER MODULE - Strategy Functions
# ======================================================
def fetch_data(ticker, period="3y"):
    """Fetch historical data - using 3y to ensure we get at least 1y of usable data after indicator calculations"""
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    return df

def moving_average_crossover_strategy(df, short_window, long_window):
    """MA Crossover Strategy"""
    df = df.copy()
    df = moving_average(df, short_window, long_window)
    df['signal'] = 0
    
    # Buy when short MA crosses above long MA
    df.loc[(df['MA_short'] > df['MA_long']) & 
           (df['MA_short'].shift(1) <= df['MA_long'].shift(1)), 'signal'] = 1
    
    # Sell when short MA crosses below long MA
    df.loc[(df['MA_short'] < df['MA_long']) & 
           (df['MA_short'].shift(1) >= df['MA_long'].shift(1)), 'signal'] = -1
    
    return df

def rsi_strategy(df, period=14, oversold=30, overbought=70):
    """RSI Strategy"""
    df = df.copy()
    df = rsi(df, period)
    df['signal'] = 0
    
    # Buy when RSI crosses above oversold
    df.loc[(df['RSI'] >= oversold) & (df['RSI'].shift(1) < oversold), 'signal'] = 1
    
    # Sell when RSI crosses below overbought
    df.loc[(df['RSI'] <= overbought) & (df['RSI'].shift(1) > overbought), 'signal'] = -1
    
    return df

def macd_strategy(df, fast=12, slow=26, signal=9):
    """MACD Strategy"""
    df = df.copy()
    df = macd(df, fast, slow, signal)
    df['signal'] = 0
    
    # Buy when MACD crosses above signal
    df.loc[(df['MACD'] > df['MACD_signal']) & 
           (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 1
    
    # Sell when MACD crosses below signal
    df.loc[(df['MACD'] < df['MACD_signal']) & 
           (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = -1
    
    return df

def bollinger_bands_strategy(df, window=20, num_std=2):
    """Bollinger Bands Strategy"""
    df = df.copy()
    df = bollinger(df, window, num_std)
    df['signal'] = 0
    
    # Buy when price crosses above lower band
    df.loc[(df['Close'] >= df['BB_lower']) & 
           (df['Close'].shift(1) < df['BB_lower'].shift(1)), 'signal'] = 1
    
    # Sell when price crosses below upper band
    df.loc[(df['Close'] <= df['BB_upper']) & 
           (df['Close'].shift(1) > df['BB_upper'].shift(1)), 'signal'] = -1
    
    return df

def stochastic_oscillator_strategy(df, length=14, smoothK=3, smoothD=3, oversold=20, overbought=80):
    """Slow Stochastic Oscillator Strategy"""
    df = df.copy()
    df = stochastic(df, length, smoothK, smoothD)
    df['signal'] = 0
    
    # Buy when %K crosses above %D and %K is below oversold
    df.loc[(df['%K'] > df['%D']) & 
           (df['%K'].shift(1) <= df['%D'].shift(1)) & 
           (df['%K'] < oversold), 'signal'] = 1
    
    # Sell when %K crosses below %D and %K is above overbought
    df.loc[(df['%K'] < df['%D']) & 
           (df['%K'].shift(1) >= df['%D'].shift(1)) & 
           (df['%K'] > overbought), 'signal'] = -1
    
    return df

def calculate_accuracy(df):
    """Calculate signal accuracy"""
    df = df.copy()
    df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    correct = 0
    total = 0
    
    # Check buy signals
    for idx in buy_signals.index:
        if idx in df.index:
            future_ret = df.loc[idx, 'future_return']
            if not pd.isna(future_ret) and future_ret > 0:
                correct += 1
            if not pd.isna(future_ret):
                total += 1
    
    # Check sell signals
    for idx in sell_signals.index:
        if idx in df.index:
            future_ret = df.loc[idx, 'future_return']
            if not pd.isna(future_ret) and future_ret < 0:
                correct += 1
            if not pd.isna(future_ret):
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    signal_count = len(buy_signals) + len(sell_signals)
    
    return accuracy, signal_count

# ======================================================
# SIDEBAR - Module Selection
# ======================================================
st.sidebar.title("📊 IDX Trading Tool")
module = st.sidebar.radio(
    "Select Module",
    ["🔍 Signal Scanner", "🎯 Strategy Optimizer"]
)

# ======================================================
# MODULE 1: SIGNAL SCANNER
# ======================================================
if module == "🔍 Signal Scanner":
    st.title("🔍 IDX Buy Signal Scanner")
    st.markdown("Scan multiple IDX stocks for buy/sell signals using technical indicators")
    
    # Indicator selector
    indicator = st.selectbox(
        "📌 Select Buy Signal Indicator",
        ["Stochastic Slow", "RSI", "MACD", "Bollinger Bands", "MA Crossover"]
    )
    
    # Inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tickers_input = st.text_area(
            "Enter IDX tickers (one per line, without .JK)",
            placeholder="BBCA\nBMRI\nTLKM\nASII\nICBP",
            height=150
        )
    
    with col2:
        st.markdown("### Indicator Parameters")
        
        if indicator == "Stochastic Slow":
            length = st.number_input("Length", 5, 30, 10)
            smoothK = st.number_input("Smooth K", 1, 10, 5)
            smoothD = st.number_input("Smooth D", 1, 10, 5)
        
        elif indicator == "RSI":
            rsi_period = st.number_input("RSI Period", 5, 30, 14)
            rsi_oversold = st.number_input("Oversold Level", 10, 40, 30)
            rsi_overbought = st.number_input("Overbought Level", 60, 90, 70)
        
        elif indicator == "MACD":
            macd_fast = st.number_input("Fast EMA", 5, 20, 12)
            macd_slow = st.number_input("Slow EMA", 20, 50, 26)
            macd_signal = st.number_input("Signal EMA", 5, 20, 9)
        
        elif indicator == "Bollinger Bands":
            bb_length = st.number_input("BB Length", 10, 40, 20)
            bb_std = st.number_input("Std Deviation", 1.0, 3.0, 2.0, 0.5)
        
        elif indicator == "MA Crossover":
            ma_short = st.number_input("Short MA Period", 3, 20, 5)
            ma_long = st.number_input("Long MA Period", 10, 100, 20)
    
    run_btn = st.button("🚀 Run Scan", type="primary", use_container_width=True)
    
    # Run scan
    if run_btn and tickers_input.strip():
        tickers = [
            t.strip().upper() + ".JK"
            for t in tickers_input.splitlines()
            if t.strip()
        ]
        
        progress = st.progress(0)
        status = st.empty()
        
        rows = []
        cache = {}
        
        for i, ticker in enumerate(tickers, 1):
            status.text(f"Scanning {i}/{len(tickers)} : {ticker}")
            progress.progress(i / len(tickers))
            
            try:
                df = yf.Ticker(ticker).history(period="1y", interval="1d")
                if df.empty:
                    continue
                
                df["PctChange"] = df["Close"].pct_change() * 100
                df["BuySignal"] = False
                df["SellSignal"] = False
                
                # Indicator logic
                if indicator == "Stochastic Slow":
                    df = stochastic(df, length, smoothK, smoothD)
                    buy_cond = (
                        (df["%K"].shift(1) < df["%D"].shift(1)) &
                        (df["%K"] > df["%D"]) &
                        (df["%K"] < 20)
                    )
                    sell_cond = (
                        (df["%K"].shift(1) > df["%D"].shift(1)) &
                        (df["%K"] < df["%D"]) &
                        (df["%K"] > 80)
                    )
                
                elif indicator == "RSI":
                    df = rsi(df, rsi_period)
                    buy_cond = (df["RSI"].shift(1) < rsi_oversold) & (df["RSI"] >= rsi_oversold)
                    sell_cond = (df["RSI"].shift(1) > rsi_overbought) & (df["RSI"] <= rsi_overbought)
                
                elif indicator == "MACD":
                    df = macd(df, macd_fast, macd_slow, macd_signal)
                    buy_cond = (
                        (df["MACD"].shift(1) < df["MACD_signal"].shift(1)) &
                        (df["MACD"] > df["MACD_signal"])
                    )
                    sell_cond = (
                        (df["MACD"].shift(1) > df["MACD_signal"].shift(1)) &
                        (df["MACD"] < df["MACD_signal"])
                    )
                
                elif indicator == "Bollinger Bands":
                    df = bollinger(df, bb_length, bb_std)
                    buy_cond = (
                        (df["Close"].shift(1) < df["BB_lower"].shift(1)) &
                        (df["Close"] >= df["BB_lower"])
                    )
                    sell_cond = (
                        (df["Close"].shift(1) > df["BB_upper"].shift(1)) &
                        (df["Close"] <= df["BB_upper"])
                    )
                
                elif indicator == "MA Crossover":
                    df = moving_average(df, ma_short, ma_long)
                    buy_cond = (
                        (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) &
                        (df["MA_short"] > df["MA_long"])
                    )
                    sell_cond = (
                        (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) &
                        (df["MA_short"] < df["MA_long"])
                    )
                
                df.loc[buy_cond, "BuySignal"] = True
                df.loc[sell_cond, "SellSignal"] = True
                
                last_buy = df[buy_cond].index[-1] if buy_cond.any() else pd.NaT
                
                cache[ticker] = df
                
                rows.append({
                    "Ticker": ticker.replace(".JK", ""),
                    "LastBuySignal": last_buy,
                    "Volume": int(df["Volume"].iloc[-1]),
                    "Close": df["Close"].iloc[-1]
                })
            
            except Exception as e:
                st.warning(f"Failed to fetch {ticker}: {str(e)}")
                continue
        
        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("LastBuySignal", ascending=False)
        
        st.session_state.scanner_result_df = result_df
        st.session_state.scanner_data_cache = cache
        st.session_state.scanner_scanned = True
        
        status.text("✅ Scan completed")
        progress.progress(1.0)
    
    # Results + Chart
    if st.session_state.scanner_scanned:
        st.markdown("---")
        st.subheader("📊 Scan Results")
        
        df = st.session_state.scanner_result_df
        
        show_only_buy = st.checkbox("Show only stocks with buy signals", True)
        if show_only_buy:
            df = df[df["LastBuySignal"].notna()]
        
        st.dataframe(df, use_container_width=True, height=300)
        
        if not df.empty:
            st.markdown("---")
            st.subheader("📈 Chart Analysis")
            
            selected = st.selectbox("Select ticker to view chart", df["Ticker"])
            dfc = st.session_state.scanner_data_cache[selected + ".JK"]
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=dfc.index,
                open=dfc["Open"],
                high=dfc["High"],
                low=dfc["Low"],
                close=dfc["Close"],
                customdata=dfc["PctChange"],
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>"
                    "Open: %{open}<br>"
                    "High: %{high}<br>"
                    "Low: %{low}<br>"
                    "Close: %{close} (%{customdata:+.2f}%)"
                    "<extra></extra>"
                ),
                name="Price"
            ), row=1, col=1)
            
            # Buy/Sell markers
            buy_df = dfc[dfc["BuySignal"]]
            sell_df = dfc[dfc["SellSignal"]]
            
            fig.add_trace(go.Scatter(
                x=buy_df.index,
                y=buy_df["Low"] * 0.995,
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="green"),
                name="Buy"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=sell_df.index,
                y=sell_df["High"] * 1.005,
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name="Sell"
            ), row=1, col=1)
            
            # Indicator panel
            if indicator == "Stochastic Slow":
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["%K"], name="%K"), row=2, col=1)
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["%D"], name="%D"), row=2, col=1)
                fig.add_hline(y=20, row=2, col=1, line_dash="dash", line_color="green")
                fig.add_hline(y=80, row=2, col=1, line_dash="dash", line_color="red")
            
            elif indicator == "RSI":
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["RSI"], name="RSI"), row=2, col=1)
                fig.add_hline(y=rsi_oversold, row=2, col=1, line_dash="dash", line_color="green")
                fig.add_hline(y=rsi_overbought, row=2, col=1, line_dash="dash", line_color="red")
            
            elif indicator == "MACD":
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MACD"], name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MACD_signal"], name="Signal"), row=2, col=1)
                fig.add_trace(go.Bar(x=dfc.index, y=dfc["MACD_hist"], name="Histogram"), row=2, col=1)
            
            elif indicator == "Bollinger Bands":
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["BB_upper"], name="BB Upper", 
                                        line=dict(dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["BB_lower"], name="BB Lower", 
                                        line=dict(dash="dot")), row=1, col=1)
            
            elif indicator == "MA Crossover":
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MA_short"], name=f"MA {ma_short}", 
                                        line=dict(color="blue")), row=1, col=1)
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MA_long"], name=f"MA {ma_long}", 
                                        line=dict(color="orange")), row=1, col=1)
            
            fig.update_layout(
                height=720,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                title=f"{selected} - {indicator}"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ======================================================
# MODULE 2: STRATEGY OPTIMIZER
# ======================================================
elif module == "🎯 Strategy Optimizer":
    st.title("🎯 Strategy Optimizer")
    st.markdown("Find the best technical indicator strategy for a specific stock")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Enter IDX Ticker (without .JK)",
            placeholder="BBCA",
            value="BBCA"
        )
    
    with col2:
        min_signal_count = st.number_input(
            "Minimum Signal Count",
            min_value=5,
            max_value=100,
            value=25,
            help="Minimum number of buy/sell signals required"
        )
    
    optimize_btn = st.button("⚡ Optimize Strategies", type="primary", use_container_width=True)
    
    if optimize_btn and ticker_input.strip():
        ticker = ticker_input.strip().upper() + ".JK"
        
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                df = fetch_data(ticker, period="2y")
                
                if df.empty:
                    st.error(f"No data available for {ticker}")
                    st.stop()
                
                st.success(f"✅ Fetched {len(df)} days of data")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_acc = -1
        best_signals = 0
        best_df = None
        best_strategy = None
        best_params = None
        results = []
        
        total_tests = 5 + 5 + 5 + 5 + 5  # Total number of parameter combinations
        current_test = 0
        
        # MA Crossover
        status_text.text("Testing MA Crossover strategies...")
        ma_params = [(3,10), (5,20), (7,25), (10,30), (15,50)]
        for short_w, long_w in ma_params:
            df_ma = moving_average_crossover_strategy(df, short_w, long_w)
            acc, signal_count = calculate_accuracy(df_ma)
            results.append(('MA Crossover', f'short={short_w}, long={long_w}', acc, signal_count, df_ma))
            if signal_count >= min_signal_count and acc > best_acc:
                best_acc = acc
                best_signals = signal_count
                best_df = df_ma
                best_strategy = 'MA Crossover'
                best_params = f'short={short_w}, long={long_w}'
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # RSI
        status_text.text("Testing RSI strategies...")
        rsi_periods = [7, 10, 14, 20, 30]
        for period in rsi_periods:
            df_rsi = rsi_strategy(df, period)
            acc, signal_count = calculate_accuracy(df_rsi)
            results.append(('RSI', f'period={period}', acc, signal_count, df_rsi))
            if signal_count >= min_signal_count and acc > best_acc:
                best_acc = acc
                best_signals = signal_count
                best_df = df_rsi
                best_strategy = 'RSI'
                best_params = f'period={period}'
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # MACD
        status_text.text("Testing MACD strategies...")
        macd_params = [(12,26,9), (5,35,5), (8,17,9), (10,40,15), (15,30,10)]
        for fast, slow, signal in macd_params:
            df_macd = macd_strategy(df, fast, slow, signal)
            acc, signal_count = calculate_accuracy(df_macd)
            results.append(('MACD', f'fast={fast}, slow={slow}, signal={signal}', acc, signal_count, df_macd))
            if signal_count >= min_signal_count and acc > best_acc:
                best_acc = acc
                best_signals = signal_count
                best_df = df_macd
                best_strategy = 'MACD'
                best_params = f'fast={fast}, slow={slow}, signal={signal}'
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # Bollinger Bands
        status_text.text("Testing Bollinger Bands strategies...")
        boll_params = [(14,2), (20,2), (10,2.5), (15,3), (20,1.5)]
        for window, num_std in boll_params:
            df_boll = bollinger_bands_strategy(df, window, num_std)
            acc, signal_count = calculate_accuracy(df_boll)
            results.append(('Bollinger Bands', f'window={window}, std={num_std}', acc, signal_count, df_boll))
            if signal_count >= min_signal_count and acc > best_acc:
                best_acc = acc
                best_signals = signal_count
                best_df = df_boll
                best_strategy = 'Bollinger Bands'
                best_params = f'window={window}, std={num_std}'
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # Stochastic (Slow)
        status_text.text("Testing Slow Stochastic strategies...")
        stochastic_params = [(14,3,3), (10,3,3), (5,3,3), (14,5,5), (20,5,5)]
        for length, smoothK, smoothD in stochastic_params:
            df_stoch = stochastic_oscillator_strategy(df, length, smoothK, smoothD, oversold=20, overbought=80)
            acc, signal_count = calculate_accuracy(df_stoch)
            results.append(('Slow Stochastic', f'Length={length}, K={smoothK}, D={smoothD}', acc, signal_count, df_stoch))
            if signal_count >= min_signal_count and acc > best_acc:
                best_acc = acc
                best_signals = signal_count
                best_df = df_stoch
                best_strategy = 'Slow Stochastic'
                best_params = f'Length={length}, K={smoothK}, D={smoothD}'
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        status_text.text("✅ Optimization completed!")
        progress_bar.progress(1.0)
        
        # Store results
        st.session_state.optimizer_results = results
        st.session_state.optimizer_best_df = best_df
        st.session_state.optimizer_best_strategy = best_strategy
        st.session_state.optimizer_best_params = best_params
        st.session_state.optimizer_best_acc = best_acc
        st.session_state.optimizer_best_signals = best_signals
        st.session_state.optimizer_ticker = ticker
        st.session_state.optimizer_min_signals = min_signal_count
    
    # Display results
    if st.session_state.optimizer_results is not None:
        st.markdown("---")
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Strategy", st.session_state.optimizer_best_strategy)
        with col2:
            st.metric("Accuracy", f"{st.session_state.optimizer_best_acc:.1%}")
        with col3:
            st.metric("Signal Count", st.session_state.optimizer_best_signals)
        with col4:
            st.metric("Parameters", st.session_state.optimizer_best_params)
        
        st.markdown("---")
        st.subheader("📊 All Strategy Results")
        
        # Create results dataframe
        results_data = []
        for strat, params, acc, sig_cnt, _ in st.session_state.optimizer_results:
            if sig_cnt >= st.session_state.optimizer_min_signals:
                results_data.append({
                    "Strategy": strat,
                    "Parameters": params,
                    "Accuracy": f"{acc:.2%}",
                    "Signals": sig_cnt
                })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values("Accuracy", ascending=False)
        st.dataframe(results_df, use_container_width=True, height=300)
        
        # Plot best strategy
        if st.session_state.optimizer_best_df is not None:
            st.markdown("---")
            st.subheader(f"📈 Best Strategy Chart: {st.session_state.optimizer_best_strategy}")
            
            best_df = st.session_state.optimizer_best_df
            ticker = st.session_state.optimizer_ticker
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=best_df.index,
                open=best_df['Open'],
                high=best_df['High'],
                low=best_df['Low'],
                close=best_df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                name=ticker
            ))
            
            # Buy signals
            buys = best_df[best_df['signal'] == 1]
            fig.add_trace(go.Scatter(
                x=buys.index,
                y=buys['Low'] * 0.995,
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=12),
                name='Buy Signal'
            ))
            
            # Sell signals
            sells = best_df[best_df['signal'] == -1]
            fig.add_trace(go.Scatter(
                x=sells.index,
                y=sells['High'] * 1.005,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12),
                name='Sell Signal'
            ))
            
            fig.update_layout(
                title=f'{ticker} - {st.session_state.optimizer_best_strategy} ({st.session_state.optimizer_best_params})<br>Accuracy: {st.session_state.optimizer_best_acc:.2%} | Signals: {st.session_state.optimizer_best_signals}',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No strategy met the minimum signal count of {st.session_state.optimizer_min_signals}")
    
    # Instructions
    with st.expander("ℹ️ How to use Strategy Optimizer"):
        st.markdown("""
        ### Strategy Optimizer Guide
        
        1. **Enter a ticker** (without .JK suffix, e.g., BBCA, BMRI, TLKM)
        2. **Set minimum signal count** - strategies with fewer signals will be excluded
        3. **Click Optimize** - the tool will test 25 different parameter combinations across 5 strategies
        
        **Strategies tested:**
        - **MA Crossover**: Tests 5 different moving average combinations
        - **RSI**: Tests 5 different RSI periods
        - **MACD**: Tests 5 different MACD parameter sets
        - **Bollinger Bands**: Tests 5 different BB configurations
        - **Stochastic**: Tests 5 different Stochastic settings
        
        **Accuracy Calculation:**
        - Checks if buy signals lead to price increases (5 days forward)
        - Checks if sell signals lead to price decreases (5 days forward)
        - Higher accuracy = more reliable signals
        
        **Note:** Past performance doesn't guarantee future results!
        """)
