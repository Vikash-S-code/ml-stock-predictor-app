import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error


TICKER = "AAPL"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


data = yf.Ticker(TICKER).history(period="10y")

data['Close_t-1'] = data['Close'].shift(1)
data['Return_t-1'] = (data['Close'] - data['Close_t-1']) / data['Close_t-1']
data['SMA_5'] = ta.sma(data['Close'], length=5)
data['RSI_14'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data['Close_t+1'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Close_t-1', 'Return_t-1', 'SMA_5', 'RSI_14', 'MACD']]
y = data['Close_t+1']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler, open(f"{MODEL_DIR}/scaler.pkl", "wb"))


models = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=10, random_state=42
    ),
    "gradient_boost": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
    "svr": SVR(C=100, gamma=0.1)
}


print("\nMODEL PERFORMANCE\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"{name.upper()} | R2:",
          round(r2_score(y_test, preds), 3),
          "| MAE:",
          round(mean_absolute_error(y_test, preds), 2))

    pickle.dump(model, open(f"{MODEL_DIR}/{name}.pkl", "wb"))

print("\nAll models saved successfully!")
