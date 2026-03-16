import pandas as pd

data = pd.read_csv("data/final_dataset.csv")

latest = data.iloc[-1]

fire = latest["fire_intensity"]
rain = latest["PRCP"]
temperature = latest["temperature"]

print("Latest Pollution Conditions")
print("Fire:", fire)
print("Rain:", rain)
print("Temperature:", temperature)

print("\nRecommended Actions:")

if fire > 1000:
    print("- Reduce crop burning activity")

if rain == 0:
    print("- Artificial rain or dust suppression recommended")

if temperature > 35:
    print("- Restrict heavy traffic during peak hours")

if fire < 100 and rain > 5:
    print("- Pollution risk low today")