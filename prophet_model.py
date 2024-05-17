import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import sqlite3

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

# Prepare data for Prophet
prophet_data = data.rename(columns={'datetime': 'ds', 'consumption': 'y'})

# Split the data into training and testing sets
train_size = int(len(prophet_data) * 0.8)
train, test = prophet_data[:train_size], prophet_data[train_size:]

# Fit Prophet model
prophet_model = Prophet(seasonality_mode='multiplicative')
prophet_model.fit(train)

# Forecast
future = prophet_model.make_future_dataframe(periods=len(test), freq='H')
forecast = prophet_model.predict(future)
prophet_forecast = forecast[-len(test):]['yhat']

# Calculate metrics
mse_prophet = mean_squared_error(test['y'], prophet_forecast)
rmse_prophet = mse_prophet ** 0.5
mae_prophet = mean_absolute_error(test['y'], prophet_forecast)
mape_prophet = (np.mean(np.abs((test['y'] - prophet_forecast) / test['y'])) * 100)

# Print metrics
print('Prophet Model:')
print('RMSE:', rmse_prophet)
print('MAE:', mae_prophet)
print('MAPE:', mape_prophet)

# Plot the actual vs forecasted values
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(test['ds'], test['y'], label='Actual')
ax.plot(test['ds'], prophet_forecast, label='Prophet Forecast')
ax.legend()
ax.set_title('Actual vs Prophet Forecasted Consumption')
plt.show()

# Save data to SQLite
conn = sqlite3.connect('energy_consumption.db')

# Save the processed data
data.to_sql('processed_data', conn, if_exists='replace', index=True)

# Save the Prophet results
results_df = pd.DataFrame({
    'actual': test['y'],
    'forecast': prophet_forecast,
    'datetime': test['ds']
})
results_df.to_sql('prophet_results', conn, if_exists='replace', index=False)

# Save model performance metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'MAPE'],
    'value': [rmse_prophet, mae_prophet, mape_prophet]
})
metrics_df.to_sql('prophet_metrics', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data and results successfully saved to SQLite database.")
