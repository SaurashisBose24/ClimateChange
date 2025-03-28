import pandas as pd
import numpy as np

# Load dataset
file_path = "./backend/GlobalTemperature.csv"
df = pd.read_csv(file_path)

# Replace '***' or similar missing values with NaN
df.replace("***", np.nan, inplace=True)  # Replace missing values
df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')  # Convert anomalies to numeric

# Rename columns for clarity
df_cleaned = df[['Year', 'J-D']].rename(columns={'J-D': 'Temperature_Anomaly'})

# Filter for baseline period (1951-1980)
baseline_period = df_cleaned[(df_cleaned['Year'] >= 1880) & (df_cleaned['Year'] <= 2024)]

# Compute the baseline average temperature anomaly
baseline_temperature = baseline_period['Temperature_Anomaly'].mean()

print(f"🌡️ The baseline temperature anomaly (1951-1980) is: {baseline_temperature:.4f}°C")

# Convert all columns (except 'Year') to numeric
for col in df.columns:
    if col != "Year":
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Select 'Year' and 'J-D' (Annual Temperature Anomaly)
df_cleaned_new = df[['Year', 'J-D']].rename(columns={'J-D': 'Temperature_Anomaly'})

# Drop missing values
df_cleaned_new.dropna(inplace=True)

# Convert 'Year' to datetime and set as index
df_cleaned_new['Year'] = pd.to_datetime(df_cleaned_new['Year'], format='%Y')
df_cleaned_new.set_index('Year', inplace=True)

# Convert Temperature Anomaly column to float
df_cleaned_new['Temperature_Anomaly'] = df_cleaned_new['Temperature_Anomaly'].astype(float)

# Check data types
print(df_cleaned_new.dtypes)

# Display cleaned data
print(df_cleaned_new.head())

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_cleaned_new[['Temperature_Anomaly']])

# Prepare data for LSTM
X, y = [], []
time_steps = 10  # Use past 10 years to predict the next
for i in range(len(df_scaled) - time_steps):
    X.append(df_scaled[i:i + time_steps])
    y.append(df_scaled[i + time_steps])

X, y = np.array(X), np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
future_steps = 20
# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict future values
# Predict future values
future_inputs = df_scaled[-time_steps:].reshape(1, time_steps, 1)  # Ensure 3D shape
future_preds = []

for _ in range(future_steps):
    pred = model.predict(future_inputs)  # Output is 2D (batch_size, 1)

    # Append prediction to future_preds
    future_preds.append(pred[0, 0])

    # Reshape and update future_inputs correctly
    pred_reshaped = np.array(pred).reshape(1, 1, 1)  # Convert to (1,1,1) shape
    future_inputs = np.concatenate((future_inputs[:, 1:, :], pred_reshaped), axis=1)  # Maintain 3D shape

# Inverse transform predictions
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Generate future year range
future_years = pd.date_range(df_cleaned_new.index[-1], periods=future_steps+1, freq='Y')[1:]

# Plot LSTM predictions
plt.figure(figsize=(10, 5))
plt.plot(df_cleaned_new.index, df_cleaned_new['Temperature_Anomaly'], label="Historical Data", color='blue')
plt.plot(future_years, future_preds, label="Predicted (LSTM)", color='red')
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Future Global Temperature Predictions (LSTM)")
plt.legend()
plt.show()
