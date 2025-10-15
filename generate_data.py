import pandas as pd
import numpy as np

# Generate synthetic time-series data
np.random.seed(42)
n = 500

timestamps = pd.date_range("2025-10-01", periods=n, freq="min")

# Simulate signals
heart_rate = np.random.normal(75, 5, n)
spo2 = np.random.normal(97, 1.5, n)
temperature = np.random.normal(36.5, 0.3, n)

# Add random anomalies
anomaly_indices = np.random.choice(n, size=10, replace=False)
heart_rate[anomaly_indices] += np.random.randint(20, 40, 10)
spo2[anomaly_indices] -= np.random.randint(5, 10, 10)
temperature[anomaly_indices] += np.random.uniform(1, 2, 10)

# Combine into DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "heart_rate": heart_rate,
    "spo2": spo2,
    "temperature": temperature
})

# Save the real CSV
df.to_csv("data/synthetic_data.csv", index=False)
print("✅ Synthetic dataset created successfully at data/synthetic_data.csv")
