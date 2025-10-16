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

    # Testing t-tests, Kruskal-Wallis, chi-square

print("\n STATISTICAL ANALYSIS \n")

# Testing pairwise t-tests by season for numeric variables
winter, spring, summer = rat[rat["season"]=="Winter"], rat[rat["season"]=="Spring"], rat[rat["season"]=="Summer"]
for variable in ["bat_landing_number","rat_arrival_number","food_availability"]:
    if variable in rat.columns:
        print(f"\n{variable} seasonal differences:")
        for s1, s2 in [("Winter","Spring"), ("Winter","Summer"), ("Spring","Summer")]:
            a = rat[rat["season"]==s1][variable].dropna()
            b = rat[rat["season"]==s2][variable].dropna()
            if len(a)>1 and len(b)>1:
                t, p = ttest_ind(a, b, equal_var=False)
                print(f"  {s1} vs {s2}: t={t:.3f}, p={p:.4f}")

# Testing association between season and risk behaviour using chi-square
ct = pd.crosstab(bat["season_label"], bat["risk"])
chi2, pv, _, _ = stats.chi2_contingency(ct)
print(f"\nRisk-Season Association: chi2={chi2:.3f}, p={pv:.4f}")

# Testing non-parametric check across seasons for rat arrivals using Kruskal-Wallis
groups = [rat[rat["season"]==s]["rat_arrival_number"].dropna() for s in ["Winter","Spring","Summer"]]
H, p = kruskal(*groups)
print(f"Kruskal-Wallis (Rat arrivals): H={H:.3f}, p={p:.4f}")

# Plotting correlations heatmap for behavioural variables
corr_cols = ["risk","reward","bat_landing_to_food","seconds_after_rat_arrival"]
corr_matrix = bat[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Behavioral Variable Correlations"); plt.show()


# preparing features and comparing simple models

print("\n PREDICTIVE MODELING \n")

# Merging seasonal context from rat data if available
context_cols = [c for c in ["rat_arrival_number", "food_availability"] if c in rat.columns]
if context_cols:
    context = rat.groupby("season")[context_cols].mean().reset_index()
else:
    context = rat[[]].copy()
bat_ml = bat.merge(context, left_on="season_label", right_on="season", how="left")

# Encoding season label to numeric code
le = LabelEncoder()
bat_ml["season_code"] = le.fit_transform(bat_ml["season_label"])

# Selecting candidate features 
Xcols = ["risk","season_code","bat_landing_to_food","seconds_after_rat_arrival",
         "rat_arrival_number","food_availability"]
Xcols = [c for c in Xcols if c in bat_ml.columns]
if not Xcols:
    raise KeyError("No feature columns available for modeling after preprocessing")
X = bat_ml[Xcols].fillna(0)
y = bat_ml["reward"]

# Splitting train/test and standardising
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

# Training logistic regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_cv = cross_val_score(lr, Xtr_s, ytr, cv=5, scoring='roc_auc')
print(f"Logistic Regression CV AUC: {lr_cv.mean():.3f} (+/- {lr_cv.std():.3f})")
lr.fit(Xtr_s, ytr)
ypl = lr.predict(Xte_s)
yprl = lr.predict_proba(Xte_s)[:,1]

# Tuning Random Forest small grid search
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(Xtr, ytr)
print(f"Random Forest Best Params: {rf_grid.best_params_}")
rf = rf_grid.best_estimator_
ypr = rf.predict(Xte)
yprr = rf.predict_proba(Xte)[:,1]

# Tuning Gradient Boosting for comparison
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_cv = cross_val_score(gb, Xtr, ytr, cv=5, scoring='roc_auc')
print(f"Gradient Boosting CV AUC: {gb_cv.mean():.3f} (+/- {gb_cv.std():.3f})")
gb.fit(Xtr, ytr)
ypg = gb.predict(Xte)
yprg = gb.predict_proba(Xte)[:,1]

# Evaluating and printing model metrics
def evaluate_model(name, yt, yp, yproba):
    """Return and print accuracy, F1 and AUC for a model."""
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp)
    auc = roc_auc_score(yt, yproba)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f}, F1-Score: {f1:.3f}, AUC: {auc:.3f}")
    print(classification_report(yt, yp, target_names=['No Reward','Reward']))
    return acc, f1, auc

lr_metrics = evaluate_model("Logistic Regression", yte, ypl, yprl)
rf_metrics = evaluate_model("Random Forest", yte, ypr, yprr)
gb_metrics = evaluate_model("Gradient Boosting", yte, ypg, yprg)

# Drawing ROC comparison plot
fpr_lr, tpr_lr, _ = roc_curve(yte, yprl)
fpr_rf, tpr_rf, _ = roc_curve(yte, yprr)
fpr_gb, tpr_gb, _ = roc_curve(yte, yprg)
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic (AUC={lr_metrics[2]:.3f})", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_metrics[2]:.3f})", linewidth=2)
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boost (AUC={gb_metrics[2]:.3f})", linewidth=2)
plt.plot([0,1], [0,1], 'k--', label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Model Comparison: ROC Curves"); plt.legend(); plt.grid(alpha=0.3); plt.show()

# Displaying feature importances by usinf Random Forest
imp = pd.Series(rf.feature_importances_, index=Xcols).sort_values(ascending=False)
sns.barplot(x=imp, y=imp.index, palette="viridis")
plt.title("Feature Importance (Random Forest)"); plt.xlabel("Importance Score"); plt.show()



