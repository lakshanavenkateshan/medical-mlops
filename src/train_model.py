import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Load cleaned dataset
data_path = "data/cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found! Run data_clean_visualize.py first.")

df = pd.read_csv(data_path)

# Define features and target
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("✅ Model Training Complete!\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_disease_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Model and Scaler saved in 'models/' folder.")
