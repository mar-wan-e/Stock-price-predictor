import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime
import os


st.set_page_config(page_title="Stock Price Predictor")

# Title
st.title("Gold, Silver and Apple Stock Price Predictor")
st.markdown("Predict tomorrow's prices for Apple, Gold, and Silver using LSTM models")

# Model configurations
MODELS = {
    "Apple Stock (AAPL)": {
        "ticker": "AAPL",
        "model_file": "apple_model.h5",
        "icon": "🍎"
    },
    "Gold (GLD)": {
        "ticker": "GLD",
        "model_file": "gold_model.h5",
        "icon": "🥇"
    },
    "Silver (SLV)": {
        "ticker": "SLV",
        "model_file": "silver_model.h5",
        "icon": "🥈"
    }
}

# Load models (FIXED VERSION)
@st.cache_resource
def load_models():
    import os
    models = {}
    for name, config in MODELS.items():
        model_path = config["model_file"]
        try:
            if os.path.exists(model_path):
                models[name] = load_model(model_path, compile=False)
            else:
                models[name] = None
        except Exception as e:
            st.sidebar.error(f"Error loading {model_path}: {str(e)}")
            models[name] = None
    return models

models = load_models()

# Sidebar - Select stock
st.sidebar.title("Select Asset")
selected_stock = st.sidebar.radio("Choose:", list(MODELS.keys()))

config = MODELS[selected_stock]
model = models[selected_stock]

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Model Info:**
- Ticker: {config['ticker']}
- Type: LSTM Neural Network
- Sequence: 5 days
- Features: OHLCV
""")

# Main area
st.header(f"{config['icon']} {selected_stock}")

if model is None:
    st.error(f"❌ Model for {selected_stock} not found!")
    st.info("Please upload the model files to the repository.")
else:
    # Get data
    @st.cache_data(ttl=3600)
    def get_data(ticker):
        try:
            tick = yf.Ticker(ticker)
            df = tick.history(period="1y")
            if "Stock Splits" in df.columns:
                df = df.drop("Stock Splits", axis=1)
            if "Dividends" in df.columns:
                df = df.drop("Dividends", axis=1)
            return df, None
        except Exception as e:
            return None, str(e)
    
    df, error = get_data(config["ticker"])
    
    if error:
        st.error(f"Error loading data: {error}")
    elif df is not None and len(df) > 5:
        # Show recent data
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📊 Recent Price Data")
            st.dataframe(df.tail(7)[['Open', 'High', 'Low', 'Close', 'Volume']], 
                        use_container_width=True)
        
        with col2:
            st.subheader("💰 Current Stats")
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100
            
            st.metric("Current Price", f"${current_price:.2f}", 
                     f"{change:+.2f} ({pct_change:+.2f}%)")
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            st.metric("52W High", f"${df['Close'].tail(252).max():.2f}")
            st.metric("52W Low", f"${df['Close'].tail(252).min():.2f}")
        
  # ========================================
# GRAPH SECTION
# ========================================

# Create tabs for different views
tab1, tab2 = st.tabs(["📈 Line Charts", "🕯️ Candlestick Charts"])

# ========================================
# TAB 1: LINE CHARTS
# ========================================
with tab1:
    # Graph 1: 90-Day Line Chart
    st.subheader("📈 Price History - Line Chart (Last 90 Days)")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index[-90:], df['Close'].tail(90), linewidth=2, color='#1f77b4')
    ax1.fill_between(df.index[-90:], df['Close'].tail(90), alpha=0.3)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title(f'{config["ticker"]} - 90 Day Price Trend', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

# ========================================
# TAB 2: CANDLESTICK CHARTS
# ========================================
with tab2:
    import matplotlib.patches as mpatches
    
    # Graph 2: 90-Day Candlestick Chart
    st.subheader("🕯️ Price History - Candlestick Chart (Last 90 Days)")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_90 = df.tail(90)
    
    for i in range(len(df_90)):
        date = df_90.index[i]
        open_price = df_90['Open'].iloc[i]
        close_price = df_90['Close'].iloc[i]
        high_price = df_90['High'].iloc[i]
        low_price = df_90['Low'].iloc[i]
        
        # Green if up, red if down
        color = 'green' if close_price >= open_price else 'red'
        
        # High-low line
        ax2.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
        
        # Open-close box
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        ax2.add_patch(mpatches.Rectangle((date, bottom), width=0.6, height=height, 
                                         facecolor=color, edgecolor='black', linewidth=0.5))
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Price ($)', fontsize=11)
    ax2.set_title(f'{config["ticker"]} - 90 Day Candlestick Chart', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")
        
st.markdown("---")
        
        # Predict button
if st.button(f"🚀 PREDICT TOMORROW'S PRICE", type="primary", use_container_width=True):
    with st.spinner("Analyzing data and making prediction..."):
        try:
        # Prepare data
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        scaler = MinMaxScaler()
        scaler.fit_transform(data)
        
        close_scaler = MinMaxScaler()
        close_scaler.fit(df[['Close']].values)
        
        # Get last 5 days
        last_5 = data[-5:]
        last_5_scaled = scaler.transform(last_5)
        input_data = last_5_scaled.reshape(1, 5, 5)
        
        # Predict
        prediction_scaled = model.predict(input_data, verbose=0)
        tomorrow_price = close_scaler.inverse_transform(prediction_scaled)[0][0]

        today_price = df['Close'].iloc[-1]
        pred_change = tomorrow_price - today_price
        pred_pct_change = (pred_change / today_price) * 100
        
        # Display prediction
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
        st.metric("Today's Close", f"${today_price:.2f}")
        
        with col2:
        st.metric("Tomorrow's Prediction", f"${tomorrow_price:.2f}")
        
        with col3:
        st.metric("Expected Change", f"${pred_change:+.2f}")
        
        with col4:
        st.metric("Expected % Change", f"{pred_pct_change:+.2f}%")
        
        # Direction
        if pred_change > 0:
        st.success(f"📈 **Prediction: Price expected to GO UP**")
        else:
        st.error(f"📉 **Prediction: Price expected to GO DOWN**")
        
        # ========================================
        # PREDICTION CHARTS (After PREDICT button)
        # ========================================
        
        # Prediction tabs
        pred_tab1, pred_tab2 = st.tabs(["📈 Line Prediction", "🕯️ Candlestick Prediction"])
        
        # ========================================
        # PREDICTION TAB 1: LINE CHART
        # ========================================
        with pred_tab1:
            st.subheader("📊 Line Chart with Prediction (30 Days)")
            
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            # Historical line
            recent_dates = df.index[-30:]
            recent_prices = df['Close'].tail(30).values
            ax3.plot(recent_dates, recent_prices, 'o-', 
                    label='Historical Prices', color='#1f77b4', linewidth=2, markersize=4)
            
            # Tomorrow prediction
            tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
            ax3.plot([df.index[-1], tomorrow_date], 
                    [today_price, tomorrow_price],
                    'o--', label='Tomorrow Prediction', 
                    color='red', linewidth=3, markersize=12)
            
            ax3.set_title(f'{config["ticker"]} - Price Prediction (Line)', 
                        fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=11)
            ax3.set_ylabel('Price ($)', fontsize=11)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig3)

        # ========================================
        # PREDICTION TAB 2: CANDLESTICK CHART
        # ========================================
        with pred_tab2:
            st.subheader("🕯️ Candlestick Chart with Prediction (30 Days)")
            
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            df_30 = df.tail(30)
            
            # Plot candlesticks for last 30 days
            for i in range(len(df_30)):
                date = df_30.index[i]
                open_price = df_30['Open'].iloc[i]
                close_price = df_30['Close'].iloc[i]
                high_price = df_30['High'].iloc[i]
                low_price = df_30['Low'].iloc[i]
                
                color = 'green' if close_price >= open_price else 'red'
                
                # High-low line
                ax4.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
        
        # Open-close box
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        ax4.add_patch(mpatches.Rectangle((date, bottom), width=0.6, height=height, 
                                         facecolor=color, edgecolor='black', 
                                         linewidth=0.5, alpha=0.7))
    
    # Add tomorrow's prediction
    tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
    ax4.plot([df.index[-1], tomorrow_date], 
            [today_price, tomorrow_price],
            'o--', label='Tomorrow Prediction', 
            color='blue', linewidth=3, markersize=12, zorder=5)
    
    # Annotation
    ax4.annotate(f'${tomorrow_price:.2f}', 
                xy=(tomorrow_date, tomorrow_price),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Price ($)', fontsize=11)
    ax4.set_title(f'{config["ticker"]} - Candlestick Prediction', 
                fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig4)

                    
# Tomorrow
tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
ax2.plot([df.index[-1], tomorrow_date], 
        [today_price, tomorrow_price],
        'o--', label='Tomorrow Prediction', 
        color='red', linewidth=3, markersize=12)

# Styling
ax2.set_title(f'{config["ticker"]} - Price Prediction', 
            fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Price ($)', fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig2)

# Confidence note
st.info("""
**⚠️ Important Notes:**
- This is a machine learning prediction, not financial advice
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
""")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    else:
        st.warning("Not enough data available for this ticker.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.caption("Data source: Yahoo Finance")
with col3:
    st.caption("Powered by LSTM Neural Networks")
