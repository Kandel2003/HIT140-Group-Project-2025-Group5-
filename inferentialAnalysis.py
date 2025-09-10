# Importing libraries that we will need for data handling, plotting and stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the cleaned datasets we prepared earlier
cleaned_df1 = pd.read_csv("CleanDataset1.csv")
cleaned_df2 = pd.read_csv("CleanDataset2.csv")

# Convert timedelta column back if needed
if cleaned_df1["time_to_food"].dtype == "object":
    cleaned_df1["time_to_food"] = cleaned_df1["time_to_food"].apply(pd.to_timedelta, errors="coerce")

# Risk-taking vs Avoidance
print("Risk-taking vs Avoidance counts:")
print(cleaned_df1["risk"].value_counts())

sns.countplot(x="risk", data=cleaned_df1, palette="Set2")
plt.title("Avoidance (0) vs Risk-taking (1)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
max_count = cleaned_df1["risk"].value_counts().max()
plt.yticks(range(0, max_count + 51, 50))
plt.show()

# Risk-taking vs Reward
crosstab = cleaned_df1.groupby(["risk", "reward"]).size().unstack(fill_value=0)
print("\nRisk vs Reward Table:")
print(crosstab)

# Descriptive Stats by Risk Behaviour
print("\nDescriptive stats by risk behaviour:")
print(cleaned_df1.groupby("risk")[["time_to_food", "rat_duration"]].describe())

# Hesitation vs Rat Duration 
sns.scatterplot(
    x="rat_duration",
    y=cleaned_df1["time_to_food"].dt.total_seconds(),
    data=cleaned_df1,
    alpha=0.5
)
plt.title("Bat hesitation vs Rat Duration")
plt.xlabel("Rat Duration (seconds)")
plt.ylabel("Time to Food (seconds)")
plt.show()
