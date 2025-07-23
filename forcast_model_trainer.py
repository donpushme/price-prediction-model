import numpy as np
import json
import os
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib

# --- Config ---
LOOKBACK = 288  # 24 hours of 5-min data
HORIZON = 288   # Predict next 24 hours (5-min intervals)
MODEL_DIR = 'models/forcast_deltas/'
DATA_PATH = 'historical_data/hist_data_pyth.json'
EPOCHS = 100
BATCH_SIZE = 64

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
with open(DATA_PATH, 'r') as f:
    data = json.load(f)
if isinstance(data, dict):
    prices = np.array(data.get('close', list(data.values())[0]), dtype=np.float32)
else:
    prices = np.array(data, dtype=np.float32)

# --- Data Quality Checks ---
prices = prices[(prices > 0) & ~np.isnan(prices) & ~np.isinf(prices)]

# --- Feature Engineering (simple: just price for now) ---
X = []
y = []
for i in range(len(prices) - LOOKBACK - HORIZON):
    x_seq = prices[i:i+LOOKBACK]
    y_seq = prices[i+LOOKBACK:i+LOOKBACK+HORIZON] - prices[i+LOOKBACK-1]  # DELTAS
    X.append(x_seq)
    y.append(y_seq)
X = np.array(X)
y = np.array(y)

# --- Scaling ---
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# --- Reshape for LSTM ---
X_scaled = X_scaled.reshape((-1, LOOKBACK, 1))

# --- Model ---
model = Sequential()
model.add(LSTM(64, input_shape=(LOOKBACK, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(HORIZON, activation='linear'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
model.summary()

# --- Train ---
history = model.fit(
    X_scaled, y_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[],
    verbose=1
)

# --- Save ---
model.save(os.path.join(MODEL_DIR, 'lstm_deltas_model.h5'))
joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X.dump'))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.dump'))

with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
    json.dump({
        'lookback': LOOKBACK,
        'horizon': HORIZON,
        'trained_at': datetime.now().isoformat(),
        'data_path': DATA_PATH
    }, f, indent=2)

print(f"Model and scalers saved to {MODEL_DIR}")
