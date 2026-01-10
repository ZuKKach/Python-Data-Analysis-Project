# ============================================
# Employee Attrition Project
# Data Processing, EDA & Machine Learning
# Course: Data Science with Python
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# 1. Load the dataset
# --------------------------------------------

file_path = r"C:\Users\KiuAdmin\PycharmProjects\PythonFinalProject\.venv\Raw_IBM-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)
print("Dataset loaded successfully.")

# --------------------------------------------
# 2. Initial Data Quality Report
# --------------------------------------------

print("\n--- Data Quality Report ---")
print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values)

# --------------------------------------------
# 3. Handle Missing Values
# --------------------------------------------

# This dataset is relatively small and HR decisions require precision.
# Since missing values are minimal, rows with missing data are removed.

if missing_values.sum() > 0:
    df = df.dropna()
    print("\nMissing values detected and removed.")
else:
    print("\nNo missing values detected.")

print("Shape after handling missing values:", df.shape)


# 4. Remove Redundant / Constant Columns These columns provide no predictive value

redundant_columns = [
    "EmployeeCount",
    "StandardHours",
    "EmployeeNumber"
]

df = df.drop(columns=redundant_columns)
print("\nDropped redundant columns:", redundant_columns)

# --------------------------------------------
# 5. Outlier Detection (IQR Method)
# --------------------------------------------

Q1 = df["MonthlyIncome"].quantile(0.25)
Q3 = df["MonthlyIncome"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["MonthlyIncome"] < lower_bound) |
    (df["MonthlyIncome"] > upper_bound)
]

print("\nMonthlyIncome outliers detected:", len(outliers))
print("Outliers retained as valid HR observations.")

# --------------------------------------------
# 6. Encoding Binary Variables
# --------------------------------------------

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

print("\nConverted Attrition and OverTime to binary values.")

# --------------------------------------------
# 7. Feature Engineering
# --------------------------------------------

satisfaction_columns = [
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

df["TotalSatisfaction"] = df[satisfaction_columns].mean(axis=1)
df["TenureRatio"] = df["YearsAtCompany"] / df["Age"]

print("\nFeature engineering completed.")

# --------------------------------------------
# 8. Exploratory Data Analysis (EDA)
# --------------------------------------------

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True) * 100)

# Attrition distribution plot
plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

# Age vs Attrition
plt.figure()
sns.histplot(data=df, x="Age", hue="Attrition", bins=30, kde=True)
plt.title("Age Distribution by Attrition")
plt.show()

# Monthly Income vs Attrition
plt.figure()
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

# Department vs Attrition
plt.figure()
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=15)
plt.show()

# Overtime vs Attrition
plt.figure()
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Attrition vs OverTime")
plt.show()

# Correlation Heatmap
numerical_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 9. Prepare Data for Machine Learning

target = "Attrition"

# Remove remaining categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns
df_model = df.drop(columns=categorical_columns)

X = df_model.drop(columns=[target])
y = df_model[target]

# 10. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed.")

# 11. Logistic Regression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 12. Decision Tree Classifier

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# --------------------------------------------
# 13. Feature Importance (Decision Tree)
# --------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# --------------------------------------------
# 14. Final Conclusion
# --------------------------------------------

print("\n--- Final Conclusion ---")
print("Employee attrition can be predicted with reasonable accuracy.")
print("Overtime, income, satisfaction, and tenure-related variables")
print("play a key role in employee turnover.")
print("Logistic Regression provides interpretability, while Decision Tree")
print("captures complex relationships.")
