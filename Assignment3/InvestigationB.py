# Exploring Investigation B by analyzing seasonal behaviour and prediction
# Exploring seasonal patterns and training simple predictive models.
# importing libaries 
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, warnings
from scipy import stats
from scipy.stats import ttest_ind, pearsonr, kruskal
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams["font.size"] = 11


# Loading data & mapping seasons

bat = pd.read_csv("Assignment3/Cleaned_Dataset1.csv")
rat = pd.read_csv("Assignment3/Cleaned_Dataset2.csv")

def map_season(m):
    """Return season name for a month number."""
    if m in [12,1,2]: return "Winter"
    elif m in [3,4,5]: return "Spring"
    elif m in [6,7,8]: return "Summer"
    elif m in [9,10,11]: return "Autumn"
    return "Other"

# Mapping months to seasons and filtering to main seasons
if "month" in bat.columns: bat["season_label"] = bat["month"].apply(map_season)
if "month" in rat.columns: rat["season"] = rat["month"].apply(map_season)
bat = bat[bat["season_label"].isin(["Winter","Spring","Summer"])].copy()
rat = rat[rat["season"].isin(["Winter","Spring","Summer"])].copy()

print(f"Dataset sizes - Bat observations: {len(bat)}, Rat observations: {len(rat)}\n")


# Exploring: distributions, trends, correlations

print(" EXPLORATORY DATA ANALYSIS \n")

# Plotting seasonal distributions
fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.violinplot(ax=axes[0], x="season", y="bat_landing_number", data=rat, palette="muted", inner="box")
sns.swarmplot(ax=axes[0], x="season", y="bat_landing_number", data=rat, color="black", alpha=0.3, size=2)
axes[0].set_title("Bat Landing Distribution by Season (Violin + Swarm)")
sns.violinplot(ax=axes[1], x="season", y="rat_arrival_number", data=rat, palette="Set2", inner="quartile")
axes[1].set_title("Rat Arrival Distribution by Season")
plt.tight_layout(); plt.show()

# Showing faceted scatter and regression to inspect temporal patterns by season
if {"hours_after_sunset","bat_landing_number","season"}.issubset(rat.columns):
    g = sns.FacetGrid(rat, col="season", hue="season", height=4, aspect=1.2, palette="viridis")
    g.map(sns.scatterplot, "hours_after_sunset", "bat_landing_number", alpha=0.6)
    g.map(sns.regplot, "hours_after_sunset", "bat_landing_number", scatter=False, ci=95)
    g.add_legend(); g.fig.suptitle("Temporal Activity Patterns Across Seasons", y=1.02)
    plt.show()

# Plotting scatter by season to examine the relationship between bat landings and rat arrivals
if {"bat_landing_number","rat_arrival_number"}.issubset(rat.columns):
    corr, p = pearsonr(rat["bat_landing_number"], rat["rat_arrival_number"])
    print(f"Bat-Rat correlation: r={corr:.3f}, p={p:.4f}")
    fig, ax = plt.subplots(figsize=(10,6))
    for season in rat["season"].unique():
        subset = rat[rat["season"]==season]
        ax.scatter(subset["rat_arrival_number"], subset["bat_landing_number"], 
                  label=season, alpha=0.6, s=80)
    ax.set_xlabel("Rat Arrival Number"); ax.set_ylabel("Bat Landing Number")
    ax.set_title(f"Species Interaction Analysis (r={corr:.3f}, p={p:.4f})")
    ax.legend(); ax.grid(alpha=0.3); plt.show()