import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
data_path = "../data/synthetic_data.csv"
output_path = "../data/cleaned_data.csv"

print("🔹 Reading synthetic data...")
df = pd.read_csv(data_path, on_bad_lines="skip")

print("✅ Raw data loaded successfully!")

# Clean data — drop empty or invalid rows
df.dropna(inplace=True)

# Ensure numeric conversion for health features
for col in ["HeartRate", "SpO2", "BodyTemp"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

print("📊 Data cleaning complete!")

# Save cleaned data
os.makedirs("../data", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"💾 Cleaned data saved to: {output_path}")

# Optional — Visualize
if "HeartRate" in df.columns and "SpO2" in df.columns and "BodyTemp" in df.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(df["HeartRate"], label="Heart Rate")
    plt.plot(df["SpO2"], label="SpO2")
    plt.plot(df["BodyTemp"], label="Body Temperature")
    plt.legend()
    plt.title("Health Metrics Over Time")
    plt.xlabel("Record Index")
    plt.ylabel("Value")
    plt.show()
