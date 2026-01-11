# Employee Attrition Analysis
# Data Processing, EDA & Machine Learning
# Course: Data Science with Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset

file_path = r"C:\Users\KiuAdmin\PycharmProjects\PythonFinalProject\.venv\Raw_IBM-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)
print("Dataset loaded successfully.")

# 2. Basic Data Overview

print("\n--- Data Overview ---")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values)


# 3. Missing Value Handling

# The dataset does not contain missing values. If any existed, rows would be removed.
if missing_values.sum() > 0:
    df.dropna(inplace=True)
    print("\nMissing values removed.")
else:
    print("\nNo missing values found.")

print("Current dataset shape:", df.shape)

# 4. Remove Non-Informative Columns
columns_to_drop = [
    "EmployeeCount",
    "StandardHours",
    "EmployeeNumber"
]
df.drop(columns=columns_to_drop, inplace=True)
print("\nRemoved redundant columns:", columns_to_drop)


# 5 Age Validation
# Employees must be at least 18 years old and  below 90

invalid_age_rows = df[
    (df["Age"] < 18) |
    (df["Age"] > 65)
]
print("\nInvalid age records detected:", len(invalid_age_rows))

# Remove invalid age rows if any exist
if len(invalid_age_rows) > 0:
    df = df[
        (df["Age"] >= 18) &
        (df["Age"] <= 90)
    ]
    print("Invalid age records removed.")

print("Age validation completed.")

# 6. Encode Binary Variables
# -----------------------------------------------------------------------------------------------------------

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
print("\nBinary variables encoded.")

# 7. Feature Engineering
# -----------------------------------------------------------------------------------------------------

satisfaction_features = [
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

df["TotalSatisfaction"] = df[satisfaction_features].mean(axis=1)
df["TenureRatio"] = df["YearsAtCompany"] / df["Age"]

print("\nNew features created.")



# 8. Exploratory Data Analysis
# ------------------------------------------------------------------------

print("\nDescriptive Statistics")
print(df.describe())

print("\nAttrition distribution:")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True) * 100)

print("Exploratory Data Analysis", "-"*100)

# Attrition count
plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

# Age vs Attrition
sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    bins=30,
    multiple="dodge"
)
plt.title("Age Distribution by Attrition")
plt.show()

# Income vs Attrition
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
plt.title("Attrition vs Overtime")
plt.show()

# Happiness Comparison: Leavers vs Stayers
# --------------------------------------------------

happiness_cols = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "TotalSatisfaction"
]

# Reshape data for easier plotting
happiness_melted = df.melt(
    id_vars="Attrition",
    value_vars=happiness_cols,
    var_name="HappinessMetric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))
sns.boxplot(
    x="HappinessMetric",
    y="Score",
    hue="Attrition",
    data=happiness_melted
)

plt.title("Happiness Comparison: Employees Who Left vs Stayed")
plt.xlabel("Happiness Metric")
plt.ylabel("Satisfaction Score")
plt.legend(title="Attrition (0 = Stayed, 1 = Left)", loc = "upper right")
plt.xticks(rotation=20)
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


# 9. Prepare Data for Modeling
# ---------------------------------------------------------------------------------------------------------------------
print("Preparing machine to learn :)", "-"*100)

target = "Attrition"

#leave  columns which are not categorical
categorical_cols = df.select_dtypes(include=["object"]).columns
df_model = df.drop(columns=categorical_cols)
#divide attrition data and other data
X = df_model.drop(columns=[target])
y = df_model[target]


# 10. Train-Test Split
# --------------------------------------==================================------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed.")

# 11. Logistic Regression Model
# --------------------------------------------------

log_model = LogisticRegression(max_iter=12923)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n Logistic Regression Results ", "-"*100)
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


# 12. Decision Tree Model
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("Decision Tree Results", "-"*50)
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))


# 13. Feature Importance (Decision Tree)
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))


# --------------------------------------------------
# 14. Conclusion

print("\n - Final Conclusion")
print("Employee attrition can be predicted with reasonable accuracy.")
print("Overtime, income, satisfaction, and tenure-related factors have strong influence on employee turnover.")
print("Logistic Regression is more interpretable, while Decision Tree captures non-linear relationships.")
