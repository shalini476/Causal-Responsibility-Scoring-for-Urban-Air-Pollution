import pandas as pd

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
aqi = pd.read_csv("data/aqi.csv")
fire = pd.read_csv("data/fire.csv")
weather = pd.read_csv("data/weather.csv")

print("AQI Columns:", aqi.columns)
print("Fire Columns:", fire.columns)
print("Weather Columns:", weather.columns)

# -----------------------------
# STEP 2: Fix date columns
# -----------------------------

# AQI
aqi.rename(columns={"Date":"date"}, inplace=True)
aqi["date"] = pd.to_datetime(aqi["date"], errors="coerce")

# Fire
fire.rename(columns={"acq_date":"date"}, inplace=True)
fire["date"] = pd.to_datetime(fire["date"], errors="coerce")

# Weather (important fix)
weather.rename(columns={"DATE":"date"}, inplace=True)
weather["date"] = pd.to_datetime(weather["date"].astype(str), format="%Y%m%d", errors="coerce")

print("\nDate conversion completed")

# -----------------------------
# STEP 3: Aggregate fire data
# -----------------------------
fire_daily = fire.groupby("date").agg({
    "frp":"sum"
}).reset_index()

fire_daily.rename(columns={"frp":"fire_intensity"}, inplace=True)

print("\nFire Daily Data")
print(fire_daily.head())

# -----------------------------
# STEP 4: Aggregate weather
# -----------------------------
weather_daily = weather.groupby("date").agg({
    "TMAX":"mean",
    "TMIN":"mean",
    "PRCP":"mean"
}).reset_index()

print("\nWeather Daily Data")
print(weather_daily.head())

# -----------------------------
# STEP 5: Clean AQI
# -----------------------------
aqi_clean = aqi[["date","AQI","PM2.5"]]

aqi_clean = aqi_clean.dropna(subset=["AQI"])

print("\nAQI Clean Data")
print(aqi_clean.head())

# -----------------------------
# STEP 6: Merge datasets
# -----------------------------
merged = pd.merge(aqi_clean, fire_daily, on="date", how="left")

merged = pd.merge(merged, weather_daily, on="date", how="left")

print("\nMerged Data Sample")
print(merged.head())

# -----------------------------
# STEP 7: Fill missing values
# -----------------------------
merged["fire_intensity"] = merged["fire_intensity"].fillna(0)

merged["TMAX"] = merged["TMAX"].fillna(merged["TMAX"].mean())
merged["TMIN"] = merged["TMIN"].fillna(merged["TMIN"].mean())
merged["PRCP"] = merged["PRCP"].fillna(0)

# create temperature feature
merged["temperature"] = (merged["TMAX"] + merged["TMIN"]) / 2

# -----------------------------
# STEP 8: Save final dataset
# -----------------------------
merged.to_csv("data/final_dataset.csv", index=False)

print("\nFinal dataset saved!")
print("Final dataset shape:", merged.shape)