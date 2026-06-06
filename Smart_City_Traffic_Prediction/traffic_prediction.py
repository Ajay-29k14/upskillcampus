import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------
# Load Data
# --------------------------

train = pd.read_csv("data/train_aWnotuB.csv")
test = pd.read_csv("data/test_BdBKkAj.csv")

# --------------------------
# Feature Engineering
# --------------------------

train["DateTime"] = pd.to_datetime(train["DateTime"])
test["DateTime"] = pd.to_datetime(test["DateTime"])

for df in [train, test]:

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

# --------------------------
# Features & Target
# --------------------------

features = [
    "Junction",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "Weekend"
]

X = train[features]
y = train["Vehicles"]

# --------------------------
# Train-Test Split
# --------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# Model Training
# --------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Validation
# --------------------------

predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5

r2 = r2_score(
    y_valid,
    predictions
)

print("\nMODEL PERFORMANCE")
print("-" * 40)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

# --------------------------
# Train on Full Dataset
# --------------------------

model.fit(X, y)

# --------------------------
# Predict Test Dataset
# --------------------------

X_test = test[features]

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "ID": test["ID"],
    "Vehicles": test_predictions
})

submission.to_csv(
    "predictions.csv",
    index=False
)

print("\nPredictions file created successfully!")

import matplotlib.pyplot as plt

# Traffic by Hour

hourly = train.groupby("Hour")["Vehicles"].mean()

plt.figure(figsize=(10,5))
hourly.plot()

plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")

plt.savefig("traffic_by_hour.png")
plt.close()

# Traffic by Junction

junction = train.groupby("Junction")["Vehicles"].mean()

plt.figure(figsize=(8,5))
junction.plot(kind="bar")

plt.title("Average Traffic by Junction")
plt.xlabel("Junction")
plt.ylabel("Average Vehicles")

plt.savefig("traffic_by_junction.png")
plt.close()

print("Graphs generated successfully!")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    y_valid,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Vehicles")
plt.ylabel("Predicted Vehicles")
plt.title("Actual vs Predicted Traffic")

plt.savefig("actual_vs_predicted.png")

plt.close()