#Importing the libraries that we will need for data handling and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, shapiro
from math import sqrt

# Loading cleaned dataset, prepared earlier 
cleaned_df1 = pd.read_csv("CleanDataset1.csv")
cleaned_df1["time_to_food"] = cleaned_df1["time_to_food"].apply(pd.to_timedelta, errors="coerce")

# 1. Chi-square Test: Risk-taking vs Reward
print("\nChi-square Test: Risk vs Reward")

# Contingency table

crosstab = cleaned_df1.groupby(["risk", "reward"]).size().unstack(fill_value=0)
chi2, p, dof, expected = chi2_contingency(crosstab)

print("Contingency Table:")
print(crosstab)
print(f"chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}")

# Effect size: Cramér's V
n = crosstab.sum().sum()
cramers_v = sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
print(f"Cramér's V = {cramers_v:.3f} (effect size)")

if p < 0.05:
    print("Significant relationship: Risk-taking behaviour is linked to rewards.")
else:
    print("No significant relationship between risk-taking and rewards.")

# Plot: Risk vs Reward counts
sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues")
plt.title("Risk-taking vs Reward (Counts)")
plt.show()


# 2. T-test: Hesitation when rats stay longer vs shorter
print("\nT-test: Hesitation vs Rat Duration")

# Split groups
median_duration = cleaned_df1["rat_duration"].median()
cleaned_df1["rat_group"] = np.where(cleaned_df1["rat_duration"] <= median_duration, "Short Rat Presence", "Long Rat Presence")

short_rats = cleaned_df1[cleaned_df1["rat_group"] == "Short Rat Presence"]["time_to_food"].dropna().dt.total_seconds()
long_rats = cleaned_df1[cleaned_df1["rat_group"] == "Long Rat Presence"]["time_to_food"].dropna().dt.total_seconds()

# Normality check
shapiro_short = shapiro(short_rats.sample(min(len(short_rats), 500)))
shapiro_long = shapiro(long_rats.sample(min(len(long_rats), 500)))

print(f"Shapiro-Wilk (Short rats): p = {shapiro_short.pvalue:.4f}")
print(f"Shapiro-Wilk (Long rats): p = {shapiro_long.pvalue:.4f}")

if shapiro_short.pvalue < 0.05 or shapiro_long.pvalue < 0.05:
    print("Normality assumption may be violated (consider Mann–Whitney U).")

# Independent t-test
t_stat, p_val = ttest_ind(short_rats, long_rats, equal_var=False)  # Welch’s t-test
print(f"t = {t_stat:.2f}, p = {p_val:.4f}")

# Effect size: Cohen’s d
mean_diff = short_rats.mean() - long_rats.mean()
pooled_sd = sqrt(((short_rats.std() ** 2) + (long_rats.std() ** 2)) / 2)
cohens_d = mean_diff / pooled_sd
print(f"Cohen's d = {cohens_d:.3f} (effect size)")

# 95% Confidence interval for mean difference
se_diff = sqrt((short_rats.var() / len(short_rats)) + (long_rats.var() / len(long_rats)))
ci_low = mean_diff - 1.96 * se_diff
ci_high = mean_diff + 1.96 * se_diff
print(f"95% CI for mean difference = [{ci_low:.2f}, {ci_high:.2f}] seconds")

if p_val < 0.05:
    print("Significant difference: Bats hesitate longer when rats stay longer.")
else:
    print("No significant difference in hesitation times between groups.")

# Plot: Boxplot of hesitation time
sns.boxplot(x="rat_group", y=cleaned_df1["time_to_food"].dt.total_seconds(), data=cleaned_df1, palette="Set2")
plt.title("Bat Hesitation Time vs Rat Duration Group")
plt.xlabel("Rat Presence Group")
plt.ylabel("Time to Food (seconds)")
plt.show()

# Plot: Violinplot for richer distribution view
sns.violinplot(x="rat_group", y=cleaned_df1["time_to_food"].dt.total_seconds(), data=cleaned_df1, palette="Set2", inner="box")
plt.title("Distribution of Hesitation Times")
plt.xlabel("Rat Presence Group")
plt.ylabel("Time to Food (seconds)")
plt.show()