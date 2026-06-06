import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("data/crop_yield.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical columns
crop_encoder = LabelEncoder()
df["Crop"] = crop_encoder.fit_transform(df["Crop"])

state_encoder = LabelEncoder()
df["State"] = state_encoder.fit_transform(df["State"])

# Features
X = df.drop("Yield (Quintal/ Hectare)", axis=1)

# Target
y = df["Yield (Quintal/ Hectare)"]

# Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_valid)

# Metrics
mae = mean_absolute_error(y_valid, predictions)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5

r2 = r2_score(y_valid, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 40)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

import matplotlib.pyplot as plt

# Yield by Crop

crop_avg = df.groupby("Crop")["Yield (Quintal/ Hectare)"].mean()

plt.figure(figsize=(10,5))
crop_avg.plot(kind="bar")

plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Yield")

plt.tight_layout()
plt.savefig("yield_by_crop.png")
plt.close()

# Yield by State

state_avg = df.groupby("State")["Yield (Quintal/ Hectare)"].mean()

plt.figure(figsize=(10,5))
state_avg.plot(kind="bar")

plt.title("Average Yield by State")
plt.xlabel("State")
plt.ylabel("Yield")

plt.tight_layout()
plt.savefig("yield_by_state.png")
plt.close()

# Actual vs Predicted

plt.figure(figsize=(8,6))

plt.scatter(
    y_valid,
    predictions
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Graphs generated successfully!")