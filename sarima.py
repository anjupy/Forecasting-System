import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import sqlite3

# Load the dataset
data = pd.read_csv("energy_consumption_levels.csv")

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Cleaning: Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Seasonal Differencing (D)
data_diff = data['consumption'].diff(24).dropna()

# ADF Test for Overall Stationarity
result = adfuller(data_diff)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Seasonal ACF and PACF plots
plot_acf(data_diff, lags=50)
plot_pacf(data_diff, lags=50)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['consumption'][:train_size], data['consumption'][train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(2, 0, 3), seasonal_order=(1, 1, 1, 24))
sarima_model_fit = sarima_model.fit(disp=False)

# Forecast
sarima_forecast = sarima_model_fit.forecast(steps=len(test))

# Calculate metrics
mse_sarima = mean_squared_error(test, sarima_forecast)
rmse_sarima = np.sqrt(mse_sarima)
mae_sarima = mean_absolute_error(test, sarima_forecast)

# Print metrics
print('SARIMA Model:')
print('RMSE:', rmse_sarima)
print('MAE:', mae_sarima)

# Save processed data and results in SQLite database
conn = sqlite3.connect('energy_consumption.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS sarima_results (
    id INTEGER PRIMARY KEY,
    rmse REAL,
    mae REAL
)
''')

# Insert evaluation results
cursor.execute('''
INSERT INTO sarima_results (rmse, mae)
VALUES (?, ?)
''', (rmse_sarima, mae_sarima))

conn.commit()
conn.close()

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
plt.legend()
plt.title('Actual vs SARIMA Forecasted Consumption')
plt.show()
