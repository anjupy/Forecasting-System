import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv("energy_consumption_levels.csv")

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Cleaning: Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Create a datetime index from the year, month, day, and hour columns
data['datetime'] = pd.to_datetime({
    'year': 2016,
    'month': data['month_of_year'],
    'day': data['day_of_month'],
    'hour': data['hour_of_day']
})

# Set the datetime column as the index
data.set_index('datetime', inplace=True)

# Select relevant features and target variable
features = ['month_of_year', 'day_of_month', 'hour_of_day']
target = 'consumption'

# Normalization/Standardization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24  # Define the length of the sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(features) + 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train[:, -1], epochs=50, batch_size=32, validation_split=0.2, shuffle=False)

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions
y_test_rescaled = scaler.inverse_transform(y_test)[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]

# Calculate metrics
rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape_lstm = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

# Print metrics
print('LSTM Model:')
print('RMSE:', rmse_lstm)
print('MAE:', mae_lstm)
print('MAPE:', mape_lstm)

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(data.index[-len(y_test_rescaled):], y_test_rescaled, label='Actual')
plt.plot(data.index[-len(y_test_rescaled):], y_pred_rescaled, label='LSTM Forecast')
plt.legend()
plt.title('Actual vs LSTM Forecasted Consumption')
plt.show()

# Save data to SQLite
conn = sqlite3.connect('energy_consumption.db')

# Save the processed data
data.to_sql('processed_data', conn, if_exists='replace', index=True)

# Save the LSTM results
results_df = pd.DataFrame({
    'datetime': data.index[-len(y_test_rescaled):],
    'actual': y_test_rescaled,
    'forecast': y_pred_rescaled
})
results_df.to_sql('lstm_results', conn, if_exists='replace', index=False)

# Save model performance metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'MAPE'],
    'value': [rmse_lstm, mae_lstm, mape_lstm]
})
metrics_df.to_sql('lstm_metrics', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data and results successfully saved to SQLite database.")
