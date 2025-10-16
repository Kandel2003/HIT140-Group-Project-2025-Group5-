#  Analyzing if bats react to rats using new cleaned dataset in investigation A
# loading libraries that we need  for data handling, plotting and models
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
 
# reading the cleaned dataset
bat = pd.read_csv("Assignment3/Cleaned_Dataset1.csv")
print(" Dataset Loaded Successfully")
print(bat.info())
print(bat.describe())
 
# performing quick descriptive checks to understand the distributions
print("\n Behavioural Distributions ")
print(bat['risk'].value_counts())
print(bat['reward'].value_counts())
 
# looking at pairwise correlations for a fast sense of relationships
corr_matrix = bat[['seconds_after_rat_arrival', 'hours_after_sunset', 'risk', 'reward']].corr()
print("\nCorrelation Matrix:\n", corr_matrix)
 
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title("Correlation Heatmap: Behaviour & Timing Variables")
plt.show()

# performing feature engineering and adding simple transformed variables
bat['time_ratio'] = bat['seconds_after_rat_arrival'] / (bat['hours_after_sunset'] + 1)
bat['interaction_term'] = bat['seconds_after_rat_arrival'] * bat['hours_after_sunset']
bat['hour_squared'] = bat['hours_after_sunset'] ** 2
 
# building a baseline Linear Regression to predict bat activity timing
X = bat[['hours_after_sunset', 'time_ratio', 'interaction_term', 'hour_squared']]
y = bat['seconds_after_rat_arrival']
 
# Splitting data into train and test sets so we can evaluate generalisation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Scaling features mean 0, std 1 so coefficients and regularisation behave well
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Fitting the ordinary least squares linear model on the scaled training data
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
 
# evaluating model performance using common regression metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
 
print("\n Linear Regression Evaluation ")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")