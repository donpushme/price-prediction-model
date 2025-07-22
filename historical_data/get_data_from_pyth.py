import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import json
import time

intervals = {'240m':'240', '60m':'60', '15m':'15', '5m':'5'}
start = '2021-01-01T00:00:00'
trading_pair = 'Crypto.BTC/USD'
BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
time_interval = 30 * 24 * 60 * 60  # 30 days in seconds

# Prepare time range
start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
end_dt = datetime.now(timezone.utc)

all_closes = []
all_times = []

current_start = start_dt

while current_start < end_dt:
    current_end = min(current_start + timedelta(seconds=time_interval), end_dt)
    from_time = int(current_start.replace(tzinfo=timezone.utc).timestamp())
    to_time = int(current_end.replace(tzinfo=timezone.utc).timestamp())

    print(f"Fetching from {current_start} to {current_end}")

    response = requests.get(BASE_URL, params={
        "symbol": trading_pair,
        "resolution": intervals['5m'],
        "from": from_time,
        "to": to_time
    })
    data = response.json()

    # Check for valid data
    if "c" in data and data["c"]:
        if(len(all_closes) > 0):
            print(all_closes[-1], data["c"][0])
            print(all_times[-1], data["t"][0])
        all_closes.extend(data["c"][:-1])
        all_times.extend(data["t"][:-1])
    else:
        print(f"No data for {current_start} to {current_end}")

    # Move to next window
    current_start = current_end
    time.sleep(1)  # Be polite to the API

# Convert to numpy array
closes = np.array(all_closes, dtype=np.float32)
times = np.array(all_times, dtype=np.int64)

# Save to npy
np.save("./historical_data/hist_data_pyth.npy", {"close": closes, "time": times})

# Save to json
with open("./historical_data/hist_data_pyth.json", "w", encoding="utf-8") as f:
    json.dump({
        "close": [float(x) for x in closes],
        "time": [int(x) for x in times]
    }, f)

print("Done! Total data points:", len(closes))