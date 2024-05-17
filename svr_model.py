import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Set the datetime column as the index
data.set_index('datetime', inplace=True)

# Select relevant features and target variable
X = data.drop(['consumption'], axis=1)
y = data['consumption']

# Normalization/Standardization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVR model with grid search for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear', 'poly']
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best estimator
best_svr = grid_search.best_estimator_
print("Best SVR Parameters:", grid_search.best_params_)

# Predictions
y_pred = best_svr.predict(X_test)

# Calculate metrics
mse_svr = mean_squared_error(y_test, y_pred)
rmse_svr = mse_svr ** 0.5
mae_svr = mean_absolute_error(y_test, y_pred)
mape_svr = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print metrics
print('SVR Model:')
print('RMSE:', rmse_svr)
print('MAE:', mae_svr)
print('MAPE:', mape_svr)

# Plot the actual vs forecasted values
plt.figure(figsize=(20, 4))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='SVR Forecast')
plt.legend()
plt.title('Actual vs SVR Forecasted Consumption')
plt.show()

# Save data to SQLite
conn = sqlite3.connect('energy_consumption.db')

# Save the processed data
data.to_sql('processed_data', conn, if_exists='replace', index=True)

# Save the SVR results
results_df = pd.DataFrame({
    'actual': y_test,
    'forecast': y_pred,
    'datetime': y_test.index
})
results_df.to_sql('svr_results', conn, if_exists='replace', index=False)

# Save model performance metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'MAPE'],
    'value': [rmse_svr, mae_svr, mape_svr]
})
metrics_df.to_sql('svr_metrics', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data and results successfully saved to SQLite database.")
