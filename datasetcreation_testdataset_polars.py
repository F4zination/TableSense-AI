import uuid
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime


date_ranges = pd.date_range(start=datetime(1900, 1, 1), end=datetime(2025, 12, 31), freq="h")
locations = ["Italy", "France", "Spain", "Portugal", "Greece", "Germany", "Austria", "Norway", "Sweeden", "Poland"]
location_multiplier = [0.7, 1.2, 0.9, 0.5, 0.3, 1.0, 1.4, 1.9, 2.1, 1.4]

data = []

print("Started constructing data...")

for dt in date_ranges:
    for i, loc in enumerate(locations):
        data.append({
            "id": str(uuid.uuid4()),
            "date": dt,
            "location": loc,
            "sales": np.random.randint(low=1, high=50) * location_multiplier[i]
        })

print("Finished constructing data!")
print("Converting to dataframe...")

data = pd.DataFrame(data)

print("Finished converting to dataframe!")
print("Saving to CSV...")

data.to_csv("data.csv", index=False)

print("Saved to CSV!")
print("Done!")