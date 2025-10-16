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