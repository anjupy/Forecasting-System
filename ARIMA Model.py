"""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.preprocessing import MinMaxScaler 
warnings.filterwarnings("ignore")
#%matplotlib inline

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

# Stationarization: Apply differencing
data_diff = data_scaled.diff().dropna()  # Apply differencing to achieve stationarity

# Create a datetime index from the year, month, day, and hour columns
data['datetime'] = pd.to_datetime({
    'year': 2016,
    'month': data['month_of_year'],
    'day': data['day_of_month'],
    'hour': data['hour_of_day']
})

# Set the datetime column as the index
data.set_index('datetime', inplace=True)

# Check the first few rows
print(data.head())

# Check the data type and statistics
print(type(data))
print(data.describe())

# Compute rolling mean
data_ma = data['consumption'].rolling(window=12).mean()

# Plot the original data and rolling mean
data['consumption'].plot(figsize=(20, 4), title='Original Data')
data_ma.plot(figsize=(20, 4), title='Rolling Mean')
plt.show()

# Boxplot of the 'consumption' column
fig, ax = plt.subplots(figsize=(20, 4))
sns.boxplot(x=data['consumption'], whis=1.5, ax=ax)
plt.show()

# Distribution plot of the 'consumption' column
plt.figure(figsize=(20, 4))
sns.histplot(data['consumption'], kde=True)
plt.show()

# Histogram of the 'consumption' column
fig = data['consumption'].hist(figsize=(20, 4))
plt.show()

# Plot rolling mean again
data_ma = data['consumption'].rolling(window=12).mean()
data_ma.plot(figsize=(20, 4), title='Rolling Mean')
plt.show()

# Shift the data for lag analysis
ndata = data.shift(1)
ndata.head()
data_base = pd.concat([data['consumption'], ndata['consumption']], axis=1)
data_base.columns = ['Actual_Value', 'Forcaste_Value']
data_base.dropna(inplace=True)

print(data_base.head())

# Plot ACF and PACF
plot_acf(data['consumption'], lags=50)
plot_pacf(data['consumption'], lags=50)
plt.show()

# Split the data into training and testing sets
data_train = data['consumption'][:7008]  # Approximately first 80% for training
data_test = data['consumption'][7008:]   # Remaining 20% for testing

# Fit the ARIMA model (p,d,q)
data_model = ARIMA(data_train, order=(2, 0, 3))
data_model_fit = data_model.fit()

# Print AIC
print('AIC:', data_model_fit.aic)

# Forecast
data_forecast = data_model_fit.forecast(steps=len(data_test))

# Compute RMSE and MAE
rmse = np.sqrt(mean_squared_error(data_test, data_forecast))
mae = mean_absolute_error(data_test, data_forecast)
print('RMSE:', rmse)
print('MAE:', mae)

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(data_test.index, data_test, label='Actual')
plt.plot(data_test.index, data_forecast, label='Forecast')
plt.legend()
plt.title('Actual vs Forecasted Consumption')
plt.show()

# Grid search for ARIMA hyperparameters
p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 5)

warnings.filterwarnings("ignore")

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                train, test = data['consumption'][:7008], data['consumption'][7008:]
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(test))
                error = mean_squared_error(test, y_pred)
                print(f'ARIMA{order} RMSE = {error:.2f}')
            except Exception as e:
                print(f'ARIMA{order} failed with error: {e}')
                continue"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.preprocessing import MinMaxScaler
import sqlite3

warnings.filterwarnings("ignore")

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

# Stationarization: Apply differencing
data_diff = data_scaled.diff().dropna()  # Apply differencing to achieve stationarity

# Create a datetime index from the year, month, day, and hour columns
data['datetime'] = pd.to_datetime({
    'year': 2016,
    'month': data['month_of_year'],
    'day': data['day_of_month'],
    'hour': data['hour_of_day']
})

# Set the datetime column as the index
data.set_index('datetime', inplace=True)

# Check the first few rows
print(data.head())

# Check the data type and statistics
print(type(data))
print(data.describe())

# Compute rolling mean
data_ma = data['consumption'].rolling(window=12).mean()

# Plot the original data and rolling mean
data['consumption'].plot(figsize=(20, 4), title='Original Data')
data_ma.plot(figsize=(20, 4), title='Rolling Mean')
plt.show()

# Boxplot of the 'consumption' column
fig, ax = plt.subplots(figsize=(20, 4))
sns.boxplot(x=data['consumption'], whis=1.5, ax=ax)
plt.show()

# Distribution plot of the 'consumption' column
plt.figure(figsize=(20, 4))
sns.histplot(data['consumption'], kde=True)
plt.show()

# Histogram of the 'consumption' column
fig = data['consumption'].hist(figsize=(20, 4))
plt.show()

# Plot rolling mean again
data_ma = data['consumption'].rolling(window=12).mean()
data_ma.plot(figsize=(20, 4), title='Rolling Mean')
plt.show()

# Shift the data for lag analysis
ndata = data.shift(1)
ndata.head()
data_base = pd.concat([data['consumption'], ndata['consumption']], axis=1)
data_base.columns = ['Actual_Value', 'Forcaste_Value']
data_base.dropna(inplace=True)

print(data_base.head())

# Plot ACF and PACF
plot_acf(data['consumption'], lags=50)
plot_pacf(data['consumption'], lags=50)
plt.show()

# Split the data into training and testing sets
data_train = data['consumption'][:7008]  # Approximately first 80% for training
data_test = data['consumption'][7008:]   # Remaining 20% for testing

# Fit the ARIMA model (p,d,q)
data_model = ARIMA(data_train, order=(2, 0, 3))
data_model_fit = data_model.fit()

# Print AIC
aic = data_model_fit.aic
print('AIC:', aic)

# Forecast
data_forecast = data_model_fit.forecast(steps=len(data_test))

# Compute RMSE and MAE
rmse = np.sqrt(mean_squared_error(data_test, data_forecast))
mae = mean_absolute_error(data_test, data_forecast)
print('RMSE:', rmse)
print('MAE:', mae)

# Compute MAPE


# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(data_test.index, data_test, label='Actual')
plt.plot(data_test.index, data_forecast, label='Forecast')
plt.legend()
plt.title('Actual vs Forecasted Consumption')
plt.show()

# Grid search for ARIMA hyperparameters
p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 5)

grid_search_results = []

warnings.filterwarnings("ignore")

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                train, test = data['consumption'][:7008], data['consumption'][7008:]
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(test))
                error = mean_squared_error(test, y_pred)
                grid_search_results.append((str(order), error))  # Convert tuple to string
                print(f'ARIMA{order} RMSE = {error:.2f}')
            except Exception as e:
                print(f'ARIMA{order} failed with error: {e}')
                continue


# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((data_test - data_forecast) / data_test)) * 100

print('MAPE/Accuracy:', mape)
# Save data to SQLite
conn = sqlite3.connect('energy_consumption.db')

# Save the processed data
data.to_sql('processed_data', conn, if_exists='replace', index=True)

# Save the ARIMA results
results_df = pd.DataFrame({
    'actual': data_test,
    'forecast': data_forecast
})
results_df.to_sql('arima_results', conn, if_exists='replace', index=True)

# Save model performance metrics
metrics_df = pd.DataFrame({
    'metric': ['AIC', 'RMSE', 'MAE'],
    'value': [aic, rmse, mae]
})
metrics_df.to_sql('model_metrics', conn, if_exists='replace', index=False)

# Save grid search results
grid_search_df = pd.DataFrame(grid_search_results, columns=['order', 'rmse'])
grid_search_df.to_sql('grid_search_results', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data and results successfully saved to SQLite database.")
