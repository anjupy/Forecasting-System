import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sqlite3

# Load the dataset
data = pd.read_csv("energy_consumption_levels.csv")

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Cleaning: Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['consumption'][:train_size], data['consumption'][train_size:]

# Fit Exponential Smoothing model
ets_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=24)
ets_model_fit = ets_model.fit()

# Forecast
ets_forecast = ets_model_fit.forecast(len(test))

# Calculate metrics
mse_ets = mean_squared_error(test, ets_forecast)
rmse_ets = np.sqrt(mse_ets)
mae_ets = mean_absolute_error(test, ets_forecast)

# Print metrics
print('Exponential Smoothing (ETS) Model:')
print('RMSE:', rmse_ets)
print('MAE:', mae_ets)

# Save results in SQLite database
conn = sqlite3.connect('energy_consumption.db')
cursor = conn.cursor()

# Create table for ETS results
cursor.execute('''
CREATE TABLE IF NOT EXISTS ets_results (
    id INTEGER PRIMARY KEY,
    rmse REAL,
    mae REAL
)
''')

# Insert evaluation results
cursor.execute('''
INSERT INTO ets_results (rmse, mae)
VALUES (?, ?)
''', (rmse_ets, mae_ets))

conn.commit()
conn.close()

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, ets_forecast, label='ETS Forecast')
plt.legend()
plt.title('Actual vs ETS Forecasted Consumption')
plt.show()
