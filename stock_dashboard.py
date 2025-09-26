import os 
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from dotenv import load_dotenv
import yfinance as yf

# Load environment variables
load_dotenv()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# ----------------------------
# Fetch Stock Data
# ----------------------------
def fetch_data(symbol: str, interval: str = "1min", outputsize: str = "compact") -> pd.DataFrame:
    if ALPHAVANTAGE_API_KEY:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": ALPHAVANTAGE_API_KEY,
            "outputsize": outputsize
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            key = f"Time Series ({interval})"
            if key in data:
                df = pd.DataFrame(data[key]).T
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index = pd.to_datetime(df.index)
                df = df.astype(float).sort_index()
                return df
        except:
            st.warning("Alpha Vantage failed, using yfinance instead.")

    interval_map = {"1min": "1m", "5min": "5m", "15min": "15m"}
    yf_interval = interval_map.get(interval, "1m")
    try:
        df = yf.download(tickers=symbol, period="1d", interval=yf_interval)
        if not df.empty:
            df.rename(columns=str.lower, inplace=True)
            if isinstance(df['close'], pd.DataFrame):
                df['close'] = df['close'].iloc[:, 0]
            return df
    except:
        st.error("Failed to fetch data from yfinance.")
    return pd.DataFrame()

# ----------------------------
# Technical Indicators
# ----------------------------
def SMA(df, period=14): return df["close"].rolling(window=period).mean()
def EMA(df, period=14): return df["close"].ewm(span=period, adjust=False).mean()
def RSI(df, period=14):
    close = df["close"].values.flatten()
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period,min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1+rs))
def MACD(df, short=12,long=26,signal=9):
    close = df["close"].values.flatten()
    short_ema = pd.Series(close).ewm(span=short,adjust=False).mean()
    long_ema = pd.Series(close).ewm(span=long,adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal,adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ----------------------------
# Streamlit Light & Fully Colorful CSS
# ----------------------------
st.markdown("""
    <style>
    /* Main page background */
    .reportview-container {
        background: linear-gradient(120deg, #ffffff 0%, #f0f4c3 100%);
        color: #000000;
        font-family: 'Trebuchet MS', sans-serif;
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #ffcc80, #ffab91);
        color: #000000;
        font-weight: bold;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4db6ac;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    /* Sliders */
    .stSlider>div>div>div>div {
        background: #ff7043;
    }
    /* Headings */
    h1 {color: #f44336;}
    h2 {color: #3f51b5;}
    h3 {color: #4caf50;}
    h4 {color: #ff9800;}
    h5 {color: #9c27b0;}
    h6 {color: #009688;}
    /* Metrics cards */
    .stMetric {
        background: linear-gradient(to right, #ffeb3b, #ff9800);
        border-radius: 12px;
        padding: 10px;
        color: #000000;
        font-weight: bold;
    }
    /* Table header */
    .stDataFrame thead {
        background-color: #81d4fa;
        color: black;
    }
    /* Table rows */
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #e1f5fe;
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #b3e5fc;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Main Dashboard
# ----------------------------
def main():
    st.set_page_config(page_title="📈 Colorful Stock Dashboard", layout="wide")
    st.markdown("<h1>📊Real-Time Stock Market Dashboard 📈</h1>", unsafe_allow_html=True)

    # ----------------- Sidebar -----------------
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    interval = st.sidebar.selectbox("Interval", ["1min","5min","15min"])
    refresh_rate = st.sidebar.slider("Auto Refresh (seconds)", 10, 300, 60)
    show_sma = st.sidebar.checkbox("Show SMA", True)
    show_ema = st.sidebar.checkbox("Show EMA", False)
    show_rsi = st.sidebar.checkbox("Show RSI", True)
    show_macd = st.sidebar.checkbox("Show MACD", True)
    run_live = st.sidebar.checkbox("Run Live Updates", False)
    compare_symbols = st.sidebar.multiselect("Compare with other stocks", ["TSLA","MSFT","GOOG","AMZN"], default=[])
    download_data = st.sidebar.checkbox("Enable CSV Download", True)

    placeholder = st.empty()

    while True:
        df = fetch_data(symbol, interval)
        if df.empty:
            st.error("No data found.")
            return
        if isinstance(df["close"], pd.DataFrame):
            df["close"] = df["close"].iloc[:,0]

        # ---------------- Indicators ----------------
        if show_sma: df["SMA"]=SMA(df)
        if show_ema: df["EMA"]=EMA(df)
        if show_rsi: df["RSI"]=RSI(df)
        if show_macd:
            macd_line, signal_line, hist = MACD(df)
            df["MACD"]=pd.Series(macd_line, index=df.index)
            df["Signal"]=pd.Series(signal_line, index=df.index)
            df["Hist"]=pd.Series(hist, index=df.index)

        # ---------------- Candlestick Chart ----------------
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#4caf50", decreasing_line_color="#f44336", name="Candlestick"
        ))
        if show_sma: fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], line=dict(color="#2196f3",width=2), name="SMA"))
        if show_ema: fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], line=dict(color="#ff9800",width=2), name="EMA"))
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#9c27b0", yaxis="y2", opacity=0.4))
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
            title=f"{symbol} Prices & Volume",
            template="plotly_white"
        )

        # ----------------- Display -----------------
        with placeholder.container():
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            if show_rsi and "RSI" in df.columns:
                st.markdown("<h2>RSI (14)</h2>", unsafe_allow_html=True)
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#e91e63",width=2), name="RSI"))
                rsi_fig.add_hline(y=70,line_dash="dash",line_color="red",annotation_text="Overbought")
                rsi_fig.add_hline(y=30,line_dash="dash",line_color="green",annotation_text="Oversold")
                rsi_fig.update_layout(yaxis=dict(range=[0,100]), template="plotly_white")
                st.plotly_chart(rsi_fig, use_container_width=True)

            # MACD
            if show_macd and "MACD" in df.columns and "Signal" in df.columns:
                st.markdown("<h2>MACD</h2>", unsafe_allow_html=True)
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#00bcd4",width=2), name="MACD"))
                macd_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], line=dict(color="#ffc107",width=2), name="Signal"))
                st.plotly_chart(macd_fig, use_container_width=True)

            # Key Metrics Cards
            st.markdown("<h2>Key Indicators</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Last Close", df['close'].iloc[-1])
            col2.metric("SMA(14)", round(df['SMA'].iloc[-1],2) if "SMA" in df else "N/A")
            col3.metric("EMA(14)", round(df['EMA'].iloc[-1],2) if "EMA" in df else "N/A")
            col4.metric("RSI(14)", round(df['RSI'].iloc[-1],2) if "RSI" in df else "N/A")

            # Volume Spike Analysis
            st.markdown("<h2>Volume Spike Analysis</h2>", unsafe_allow_html=True)
            vol_threshold = df['volume'].mean() + 2*df['volume'].std()
            high_vol = df[df['volume'] > vol_threshold]
            st.bar_chart(high_vol['volume'], use_container_width=True)

            # Stock Comparison
            if compare_symbols:
                st.markdown("<h2>Stock Comparison</h2>", unsafe_allow_html=True)
                comp_df = pd.DataFrame()
                for sym in compare_symbols:
                    temp = fetch_data(sym, interval)
                    if not temp.empty:
                        if isinstance(temp['close'], pd.DataFrame):
                            temp['close'] = temp['close'].iloc[:,0]
                        comp_df[sym] = temp['close']
                if not comp_df.empty:
                    st.line_chart(comp_df, use_container_width=True)

            # Heatmap
            st.markdown("<h2>Stock Heatmap (% Change Today)</h2>", unsafe_allow_html=True)
            heat_symbols = [symbol]+compare_symbols
            changes=[]
            for sym in heat_symbols:
                data=fetch_data(sym)
                if not data.empty:
                    close = data['close'].iloc[-1]
                    open_ = data['close'].iloc[0]
                    change = (close-open_)/open_*100
                    changes.append((sym, change))
            if changes:
                heatmap_df=pd.DataFrame(changes, columns=["Symbol","%Change"])
                heatmap_df["%Change"]=pd.to_numeric(heatmap_df["%Change"], errors='coerce')
                heatmap_df=heatmap_df.dropna()
                fig_heat=go.Figure(data=go.Heatmap(
                    z=heatmap_df["%Change"],
                    x=heatmap_df["Symbol"],
                    y=["%Change"],
                    colorscale="Rainbow",
                    zmid=0,
                    text=heatmap_df["%Change"].round(2),
                    texttemplate="%{text}%"
                ))
                fig_heat.update_layout(height=250, template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)

            # Latest Raw Data
            st.markdown("<h2>Latest Data</h2>", unsafe_allow_html=True)
            st.dataframe(df.tail(20))

            # CSV Download
            if download_data:
                csv=df.to_csv().encode('utf-8')
                st.download_button("Download CSV", csv, file_name=f"{symbol}_data.csv", mime="text/csv")

        if not run_live: break
        time.sleep(refresh_rate)

if __name__=="__main__":
    main()
