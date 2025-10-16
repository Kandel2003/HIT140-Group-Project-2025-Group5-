# importing necessary libaries for data cleaning
import pandas as pd
import numpy as np
import re
from datetime import timedelta
 
# Loading the cleaning Datasets from  Assessment2 To further clean
df1 = pd.read_csv("Assignment3/CleanDataset1.csv")   
df2 = pd.read_csv("Assignment3/CleanDataset2.csv")   
 
print("Initial shapes:")
print("Dataset1:", df1.shape)
print("Dataset2:", df2.shape)
print()
 
#Converting relevent columns to datetime
datetime_cols_df1 = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
for col in datetime_cols_df1:
    df1[col] = pd.to_datetime(df1[col], errors='coerce')
 
df2['time'] = pd.to_datetime(df2['time'], errors='coerce')
 
 
#Function to change time difference to seconds
def to_seconds(x):
    try:
        td = pd.to_timedelta(x)
        return td.total_seconds()
    except Exception:
        return np.nan
 
if 'time_to_food' in df1.columns:
    df1['time_to_food_s'] = df1['time_to_food'].apply(to_seconds)
else:
    df1['time_to_food_s'] = np.nan
 
#Introducing a new column to categorise habits
def clean_habit(h):
    if pd.isna(h):
        return 'unknown'
    hs = str(h).lower().strip()
 
    # Removing coordinate like entries
  
    if ';' in hs or re.match(r'^[\d\.\-\,\s;]+$', hs):
        return 'other'
    # Mapping  categories with specific keywords.
    if 'attack' in hs or 'fight' in hs:
        return 'attack'
    if 'rat' in hs:
        return 'rat'
    if 'bat' in hs:
        return 'bat'
    if 'pick' in hs or 'eat' in hs or 'bowl' in hs:
        return 'pick'
    if 'gaze' in hs or 'look' in hs:
        return 'gaze'
    if 'no food' in hs or 'no_food' in hs:
        return 'no_food'
    return 'other'
 
df1['habit_clean'] = df1['habit'].apply(clean_habit)
 
print("Habit categories after cleaning:")
print(df1['habit_clean'].value_counts())
print()
 
 
df1['bat_during_rat'] = (
    (df1['start_time'] >= df1['rat_period_start']) &
    (df1['start_time'] <= df1['rat_period_end'])
)
 
 
df1['landing_end'] = df1['start_time'] + pd.to_timedelta(df1['bat_landing_to_food'], unit='s')
df1['overlap_start'] = df1[['start_time', 'rat_period_start']].max(axis=1)
df1['overlap_end'] = df1[['landing_end', 'rat_period_end']].min(axis=1)
df1['overlap_seconds'] = (df1['overlap_end'] - df1['overlap_start']).dt.total_seconds().clip(lower=0)
 
 
if 'season' in df1.columns:
    df1['season'] = df1['season'].astype('category')
if 'month' in df1.columns:
    df1['month'] = df1['month'].astype('category')
 
 
if 'rat_minutes' in df2.columns:
    df2['rat_presence_frac'] = df2['rat_minutes'] / (30 * 60)  
 
 
df2_periods = df2.copy()
df2_periods = df2_periods.set_index('time')
 
df1['period_start'] = df1['start_time'].dt.floor('30min')
 
 
#Merging two dataframes based on 30 mins. start time
df1 = df1.merge(
    df2[['time', 'rat_arrival_number', 'rat_minutes', 'bat_landing_number', 'rat_presence_frac']],
    left_on='period_start',
    right_on='time',
    how='left',
    suffixes=('', '_period')
)
 
# Five minutes landing time and Five minutes to food
df1['flag_long_landing'] = df1['bat_landing_to_food'] > 300  
df1['flag_long_time_to_food'] = df1['time_to_food_s'] > 300  
 
 
df1.to_csv("Assignment3/Cleaned_Dataset1.csv", index=False)
df2.to_csv("Assignment3/Cleaned_Dataset2.csv", index=False)
 
print(" Cleaning complete.")
print("Saved files: Cleaned_Dataset1.csv, Cleaned_Dataset2.csv")
 
 
print()
print("Dataset1 summary (after cleaning):")
print(df1.describe(include='all').T.head(12))
print()
print("Dataset2 summary (after cleaning):")
print(df2.describe(include='all').T.head(12))