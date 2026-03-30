import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("📈 Stock Buy/Sell Predictor")

stock_name = st.text_input("Enter Stock Symbol (e.g. TCS.NS)", "TCS.NS")

if st.button("Predict"):

    
    df = yf.download(stock_name, start="2020-01-01", end="2024-01-01")

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Daily_Return'] = df['Close'].pct_change()

    
    df['Future_Price'] = df['Close'].shift(-5)
    df['Target'] = np.where(df['Future_Price'] > df['Close'], 1, 0)

    df.dropna(inplace=True)

    
    X = df[['SMA_10', 'SMA_50', 'Daily_Return']]
    y = df['Target']

    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

   
    latest = X.tail(1)
    prediction = model.predict(latest)[0]

    signal = "BUY 📈" if prediction == 1 else "SELL 📉"

    st.subheader(f"Prediction: {signal}")

    
    st.write("Recent Data", df[['Close']].tail(10))

    
    st.line_chart(df['Close'])