"""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import sqlite3
import warnings

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

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print('RMSE:', rmse)
print('MAE:', mae)

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
''', (rmse, mae))

conn.commit()
conn.close()

# Plot training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()"""




import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import sqlite3
import numpy as np

def train_ann_and_save_model():
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

    # Build the ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Save the trained model
    model.save('ann_model.h5')

    return rmse, mae
