import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import pickle
import os


MODEL_DIR = "models"

st.set_page_config(
    page_title="Stock Prediction",
    layout="wide"
)


st.markdown("""
<style>
body {background-color:#0E1117;}
h1,h2,h3 {color:#7FFFD4;}
.stButton>button {
    background-color:#14B8A6;
    color:black;
    font-weight:bold;
    border-radius:10px;
}
.card {
    background-color:#161B22;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)



st.markdown("<h1>Stock Price Prediction</h1>", unsafe_allow_html=True)
st.caption("Using Sklearn + YFinance")


models = {
    "Linear Regression": pickle.load(open(f"{MODEL_DIR}/linear.pkl", "rb")),
    "Random Forest": pickle.load(open(f"{MODEL_DIR}/random_forest.pkl", "rb")),
    "Gradient Boosting": pickle.load(open(f"{MODEL_DIR}/gradient_boost.pkl", "rb")),
    "SVR": pickle.load(open(f"{MODEL_DIR}/svr.pkl", "rb"))
}

scaler = pickle.load(open(f"{MODEL_DIR}/scaler.pkl", "rb"))


st.sidebar.header("⚙ Settings")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")


data = yf.Ticker(ticker).history(period="10y")
data['SMA_5'] = ta.sma(data['Close'], length=5)
data['RSI_14'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data.dropna(inplace=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    close_t1 = st.number_input("Previous Close", value=float(data['Close'].iloc[-1]))
with c2:
    return_t1 = st.number_input("Previous Return", value=0.0)
with c3:
    sma_5 = st.number_input("SMA 5", value=float(data['SMA_5'].iloc[-1]))
with c4:
    rsi_14 = st.number_input("RSI 14", value=float(data['RSI_14'].iloc[-1]))
with c5:
    macd_val = st.number_input("MACD", value=float(data['MACD'].iloc[-1]))


if st.button("Predict"):
    model = models[model_name]
    X_user = np.array([[close_t1, return_t1, sma_5, rsi_14, macd_val]])
    X_user_scaled = scaler.transform(X_user)
    pred = model.predict(X_user_scaled)

    st.success(f"Predicted Next Day Close ({model_name}): ${pred[0]:.2f}")
