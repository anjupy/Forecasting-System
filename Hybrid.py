import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sqlite3

# Load the dataset
data = pd.read_csv("energy_consumption_levels.csv")

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Cleaning: Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Normalization/Standardization
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Create lag features
for i in range(1, 13):
    data_scaled[f'lag_{i}'] = data_scaled['consumption'].shift(i)

# Drop rows with NaN values
data_scaled.dropna(inplace=True)

# Split the dataset into features (X) and target variable (y)
X = data_scaled.drop(columns=['consumption'])
y = data_scaled['consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit the ARIMA model
arima_model = ARIMA(y_train, order=(2, 0, 3))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(X_test))

# ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Predict with ANN
ann_forecast = model.predict(X_test)

# Combine forecasts
hybrid_forecast = 0.5 * arima_forecast + 0.5 * ann_forecast.flatten()

# Calculate metrics
mse_hybrid = mean_squared_error(y_test, hybrid_forecast)
rmse_hybrid = np.sqrt(mse_hybrid)
mae_hybrid = mean_absolute_error(y_test, hybrid_forecast)

# Print metrics
print('Hybrid Model:')
print('RMSE:', rmse_hybrid)
print('MAE:', mae_hybrid)

# Save processed data and results in SQLite database
conn = sqlite3.connect('energy_consumption.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS processed_data (
    id INTEGER PRIMARY KEY,
    consumption REAL,
    lag_1 REAL,
    lag_2 REAL,
    lag_3 REAL,
    lag_4 REAL,
    lag_5 REAL,
    lag_6 REAL,
    lag_7 REAL,
    lag_8 REAL,
    lag_9 REAL,
    lag_10 REAL,
    lag_11 REAL,
    lag_12 REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    rmse REAL,
    mae REAL
)
''')

# Insert processed data
data_scaled.reset_index(drop=True, inplace=True)
data_scaled.to_sql('processed_data', conn, if_exists='replace', index_label='id')

# Insert evaluation results
cursor.execute('''
INSERT INTO results (rmse, mae)
VALUES (?, ?)
''', (rmse_hybrid, mae_hybrid))

conn.commit()
conn.close()

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, hybrid_forecast, label='Hybrid Forecast')
plt.legend()
plt.title('Actual vs Hybrid Forecasted Consumption')
plt.show()
