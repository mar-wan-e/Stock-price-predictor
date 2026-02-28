# 📈 Stock Price Predictor

AI-powered stock price prediction using LSTM neural networks for Apple, Gold, and Silver.

## 🚀 Live Demo

[Click here to use the app](https://your-app-url.streamlit.app)

## Features

- 🍎 Apple Stock (AAPL) prediction
- 🥇 Gold (GLD) prediction
- 🥈 Silver (SLV) prediction
- 📊 Interactive charts
- 📈 Real-time data from Yahoo Finance
- 🤖 LSTM deep learning models

## How to Use

1. Select an asset from the sidebar
2. View current price data and charts
3. Click "PREDICT TOMORROW'S PRICE"
4. See the prediction with visualization

## Technology

- **Framework**: Streamlit
- **ML Model**: LSTM (Long Short-Term Memory)
- **Data Source**: Yahoo Finance API
- **Deployment**: Streamlit Cloud

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Training

Models are trained on 10 years of historical data using:
- Sequence length: 5 days
- Features: Open, High, Low, Close, Volume
- Architecture: 3-layer LSTM with dropout

## Disclaimer

⚠️ This tool is for educational purposes only. Not financial advice.