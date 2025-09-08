import pandas as pd

# Load datasets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# === Clean dataset1.csv ===
# Convert time-related columns to datetime
time_cols1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
for col in time_cols1:
    df1[col] = pd.to_datetime(df1[col], errors="coerce")  # invalid times -> NaT

# Derive useful columns
df1["rat_duration"] = (df1["rat_period_end"] - df1["rat_period_start"]).dt.total_seconds()
df1["time_to_food"] = pd.to_timedelta(df1["bat_landing_to_food"], errors="coerce")

# handle missing values
df1["habit"] = df1["habit"].fillna("Unknown")

# Drop rows where critical time values are missing
df1 = df1.dropna(subset=["start_time", "sunset_time"])

# === Clean dataset2.csv ===
time_cols2 = ["time"]
for col in time_cols2:
    df2[col] = pd.to_datetime(df2[col], errors="coerce")

df2 = df2.dropna(subset=["time"])  # remove invalid time rows

# Reset index after cleaning
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

# Save cleaned versions
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)

print("Cleaning done!")
print(df1.head())
print(df2.head())
