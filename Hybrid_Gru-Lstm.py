import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import os
import warnings

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
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

# Load the data
data = pd.read_csv('delta.csv', parse_dates=['Date'], index_col='Date')
data['Delta_t'] = data['effr_t-1']
data['S_T'] = data['Actual'] - data['Forecast']
data['R_T'] = data['Released'] - data['Corrected']
data.dropna(inplace=True)

# Load and process stock price data
stock_data = yf.download('^GSPC', start='2007-01-01', end='2025-01-01')
stock_data['Price'] = stock_data['Adj Close'].pct_change()  # Daily returns
stock_data.dropna(inplace=True)
stock_data.index = pd.to_datetime(stock_data.index)

# Calculate EMA
ema_span = 100  # You can choose a different span
stock_data['EMA'] = stock_data['Adj Close'].ewm(span=ema_span, adjust=False).mean()

# Calculate RSI
rsi_window = 14  # You can choose a different window
stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'], window=rsi_window)

# Calculate MACD
short_window = 12
long_window = 26
signal_window = 9
stock_data['MACD'], stock_data['Signal_Line'] = calculate_macd(stock_data['Adj Close'], short_window, long_window, signal_window)

# Combine datasets based on index (date)
data = pd.concat([data, stock_data[['Price', 'EMA', 'RSI', 'MACD' ]]], axis=1, join='inner')
data.dropna(inplace=True)  # Ensure no NaN values are present

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Delta_t', 'S_T', 'R_T', 'EMA', 'RSI', 'MACD', 'Price']])

# Split data into features and target
X = data_scaled[:, :-1]
y = data_scaled[:, -1]  # Assuming 'Price' is the last column

# Reshape input to be [samples, time steps, features] for RNN
X = X.reshape(X.shape[0], 1, X.shape[1])

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(X))

# Use indices from the best fold (fold 4)
train_index, val_index = splits[3]  # Fold indices start from 0, so fold 4 is at index 3

X_train, X_val = X[train_index], X[val_index]
y_train, y_val = y[train_index], y[val_index]
warnings.filterwarnings('ignore')
# Build and train the model
model = Sequential()
model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model_checkpoint = ModelCheckpoint('best_model_fold_4.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=1, callbacks=[model_checkpoint])

best_model = tf.keras.models.load_model('best_model_fold_4.keras')

#Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='GRU Train Loss')
plt.plot(history.history['val_loss'], label='GRU Validation Loss')
plt.title('GRU-LSTM Model Loss for Best Fold (Fold 4)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Prepare test data
X_test, y_test = X[val_index], y[val_index]  # Use the validation set from the best fold as test data

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'GRU-LSTM Mean Squared Error: {mse}')

mae = mean_absolute_error(y_test, y_pred)
print(f'GRU-LSTM Mean Squared Error: {mae}')


# Prepare the most recent data point for the next prediction
most_recent_data = data.iloc[-1][['Delta_t', 'S_T', 'R_T', 'EMA', 'RSI', 'MACD', 'Price']].values.reshape(1, -1)
most_recent_data_scaled = scaler.transform(most_recent_data)

# Exclude the 'Price' column when passing the data to the model for prediction
most_recent_data_for_prediction = most_recent_data_scaled[:, :-1].reshape(1, 1, -1)

# Predict the next price change
next_price_change_scaled = best_model.predict(most_recent_data_for_prediction)

# Print scaled next price change to check the range
print(f'Scaled Next Price Change: {next_price_change_scaled}')



# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Price Changes')
plt.plot(y_pred, label='GRU-LSTM Predicted Price Changes')
plt.title('GRU-LSTM Actual vs Predicted Stock Price Changes')
plt.xlabel('Sample')
plt.plot(range(len(y_pred), len(y_pred) + 1), next_price_change_scaled, 'ro', label='Next Predicted Price Change')
plt.xticks(range(0,40))
plt.ylabel('Price Change')
plt.axhline(y=0, color='red', linestyle='--')
plt.grid(True)
plt.legend()
plt.show()
