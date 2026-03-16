import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = pd.read_csv("data/final_dataset.csv")

print("Dataset Loaded")
print(data.head())

# -----------------------------
# STEP 2: Handle Missing Values
# -----------------------------
data["fire_intensity"] = data["fire_intensity"].fillna(0)
data["PRCP"] = data["PRCP"].fillna(0)

data["TMAX"] = data["TMAX"].fillna(data["TMAX"].mean())
data["TMIN"] = data["TMIN"].fillna(data["TMIN"].mean())

# Create temperature feature
data["temperature"] = (data["TMAX"] + data["TMIN"]) / 2

# Remove rows without AQI
data = data.dropna(subset=["AQI"])

print("\nClean Dataset Size:", data.shape)

# -----------------------------
# STEP 3: Pollution Momentum Index (PMI)
# -----------------------------
data["PMI"] = data["AQI"].diff()

print("\nAQI with PMI")
print(data[["AQI", "PMI"]].head())

# -----------------------------
# STEP 4: Select Features
# -----------------------------
X = data[["fire_intensity", "temperature", "PRCP"]]
y = data["AQI"]

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: Train Model
# -----------------------------
model = RandomForestRegressor()

model.fit(X_train, y_train)

print("\nModel training completed")

# -----------------------------
# STEP 7: Prediction
# -----------------------------
pred = model.predict(X_test)

error = mean_absolute_error(y_test, pred)

print("Model MAE:", error)

# -----------------------------
# STEP 8: Feature Importance
# -----------------------------
importance = model.feature_importances_

print("\nSource Responsibility")
for name, score in zip(X.columns, importance):
    percent = round(score * 100, 2)
    print(f"{name} contributes approx {percent}% to pollution")

# -----------------------------
# STEP 9: Counterfactual Simulation
# -----------------------------
X_no_fire = X.copy()

X_no_fire["fire_intensity"] = 0

pred_no_fire = model.predict(X_no_fire)

# Responsibility score
actual_aqi = data["AQI"]

responsibility = (actual_aqi - pred_no_fire) / actual_aqi

print("\nFire Contribution Example:")
print(responsibility.head())

# -----------------------------
# STEP 10: Graph 1 - AQI Trend
# -----------------------------
plt.figure()

plt.plot(data["AQI"][:100])

plt.title("AQI Trend Over Time")
plt.xlabel("Time")
plt.ylabel("AQI")

plt.show()

# -----------------------------
# STEP 11: Graph 2 - Source Impact
# -----------------------------
plt.figure()

plt.bar(X.columns, importance)

plt.title("Pollution Source Contribution")
plt.xlabel("Source")
plt.ylabel("Importance")

plt.show()

# -----------------------------
# STEP 12: Graph 3 - Counterfactual Simulation
# -----------------------------
plt.figure()

plt.plot(actual_aqi.values[:100], label="Actual AQI")
plt.plot(pred_no_fire[:100], label="AQI Without Fire")

plt.legend()

plt.title("Counterfactual Pollution Simulation")

plt.show()