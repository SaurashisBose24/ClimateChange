from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes



# Load dataset
file_path = "./backend/GlobalTemperature.csv"
df = pd.read_csv(file_path)

df.replace("***", np.nan, inplace=True)
df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
df_cleaned_new = df[['Year', 'J-D']].rename(columns={'J-D': 'Temperature_Anomaly'})
df_cleaned_new.dropna(inplace=True)
df_cleaned_new['Year'] = pd.to_datetime(df_cleaned_new['Year'], format='%Y')
df_cleaned_new.set_index('Year', inplace=True)
df_cleaned_new['Temperature_Anomaly'] = df_cleaned_new['Temperature_Anomaly'].astype(float)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_cleaned_new[['Temperature_Anomaly']])

# Prepare data for LSTM
X, y = [], []
time_steps = 20  # Use past 10 years to predict the next
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

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

@app.route('/predict', methods=['GET'])
def predict():
    future_steps = 10
    future_inputs = df_scaled[-time_steps:].reshape(1, time_steps, 1)
    future_preds = []
    
    for _ in range(future_steps):
        pred = model.predict(future_inputs)
        future_preds.append(pred[0, 0])
        pred_reshaped = np.array(pred).reshape(1, 1, 1)
        future_inputs = np.concatenate((future_inputs[:, 1:, :], pred_reshaped), axis=1)
    
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten().tolist()
    future_years = pd.date_range(df_cleaned_new.index[-1], periods=future_steps+1, freq='Y')[1:].year.tolist()
    print("done")
    return jsonify({'years': future_years, 'predictions': future_preds})

@app.route('/historic', methods=['GET'])
def historic():
    years = df_cleaned_new.index.year.tolist()
    anomalies = df_cleaned_new['Temperature_Anomaly'].tolist()
    return jsonify({'years': years, 'anomalies': anomalies})


if __name__ == '__main__':
    app.run(debug=True)
