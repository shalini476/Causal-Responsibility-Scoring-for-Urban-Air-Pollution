import pandas as pd
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv("data/final_dataset.csv")

# AQI Trend
plt.figure()
plt.plot(data["AQI"][:200])
plt.title("AQI Trend Over Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()

# Fire intensity vs AQI
plt.figure()
plt.scatter(data["fire_intensity"], data["AQI"])
plt.title("Fire Intensity vs AQI")
plt.xlabel("Fire Intensity")
plt.ylabel("AQI")
plt.show()