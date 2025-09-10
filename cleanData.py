#Importing the pandas for data handling
import pandas as pd

# Loading the raw datasets
raw1= pd.read_csv("dataset1.csv")
raw2 = pd.read_csv("dataset2.csv")


#Cleaning the first dataset1.csv

#Converting time related columns to datetime
time_cols1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
for col in time_cols1:
# Converting the columns to datetime, invalid values become NaT
    raw1[col] = pd.to_datetime(raw1[col], errors="coerce")  

# Adding new helper columns
raw1["rat_duration"] = (raw1["rat_period_end"] - raw1["rat_period_start"]).dt.total_seconds()
raw1["time_to_food"] = pd.to_timedelta(raw1["bat_landing_to_food"], errors="coerce")


# Handling missing values
raw1["habit"] = raw1["habit"].fillna("Unknown")

# Droping rows where critical time values are missing
raw1 = raw1.dropna(subset=["start_time", "sunset_time"])

# Cleaning the second datset2.csv
time_cols2 = ["time"]
for col in time_cols2:
 # Converting to datetime, invalid values to NaT
     raw2[col] = pd.to_datetime(raw2[col], errors="coerce")

# Droping rows where critical time values are missing
raw2 = raw2.dropna(subset=["time"])  

# Reseting index of both dataset after cleaning
raw1.reset_index(drop=True, inplace=True)
raw2.reset_index(drop=True, inplace=True)

# Save cleaned versions
raw1.to_csv("CleanDataset1.csv", index=False)
raw2.to_csv("CleanDataset2.csv", index=False)

print("Cleaning done!")
print(raw1.head())
print(raw2.head())