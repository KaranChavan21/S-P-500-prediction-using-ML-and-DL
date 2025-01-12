# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import random
import os
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Load the macroeconomic data
data = pd.read_csv('delta.csv', parse_dates=['Date'], index_col='Date')
data['Delta_t'] = data['effr_t-1']
data['S_T'] = data['Actual'] - data['Forecast']
data['R_T'] = data['Released'] - data['Corrected']
data.dropna(inplace=True)

# Load and process stock price data
stock_data = yf.download('^GSPC', start='2007-01-01', end='2025-01-01')
stock_data['Price'] = stock_data['Adj Close'].pct_change()  # Daily returns
stock_data.dropna(inplace=True)

# Calculate EMA
ema_span = 100
stock_data['EMA'] = stock_data['Adj Close'].ewm(span=ema_span, adjust=False).mean()

# Calculate RSI
rsi_window = 14
stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'], window=rsi_window)

# Calculate MACD
short_window = 12
long_window = 26
signal_window = 9
stock_data['MACD'], stock_data['Signal_Line'] = calculate_macd(
    stock_data['Adj Close'], short_window, long_window, signal_window)

# Combine datasets based on index (date)
data_combined = pd.concat(
    [data, stock_data[['Price', 'EMA', 'RSI', 'MACD']]], axis=1, join='inner')
data_combined.dropna(inplace=True)  # Ensure no NaN values are present

# Create lagged features
lag_features = ['Price']
lagged_data = data_combined.copy()

for feature in lag_features:
    for lag in range(1, 6):  # Create lags 1 to 5
        lagged_data[f'{feature}_lag_{lag}'] = lagged_data[feature].shift(lag)

# Drop rows with NaN values after creating lagged features
lagged_data.dropna(inplace=True)

# Define features and target variable
features = lagged_data.drop(['Price'], axis=1)
target = lagged_data['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False)

# Initialize and fit the XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Generate predictions
predictions_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, predictions_xgb)
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
print(f'XGBoost Mean Squared Error: {mse_xgb}')
print(f'XGBoost Mean Absolute Error: {mae_xgb}')

# Predict the next price change
# Prepare the last row of data for prediction
last_row = features.iloc[-1].values.reshape(1, -1)
next_prediction_xgb = xgb_model.predict(last_row)
print(f'Predicted Next Price Change (XGBoost): {next_prediction_xgb[0]}')

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Price Changes')
plt.plot(y_test.index, predictions_xgb, label='XGBoost Predicted Price Changes', color='green')
plt.title('Actual vs. XGBoost Predicted Stock Price Changes')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.grid(True)
plt.show()
