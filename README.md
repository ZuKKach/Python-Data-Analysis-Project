# Python-Data-Analysis-Project
Identify why employees leave. Construct a model that predicts employee attrition.

Author: Zurabi Katcharava

## Project Overview
This project analyzes employee attrition from the IBM HR Analytics dataset.  
The goal is to understand key factors influencing employee turnover and build predictive machine learning models.

The project covers:
- Data cleaning and preprocessing
- Exploratory Data Analysis 
- Classification using Logistic Regression and Decision Tree

## Dataset
- Source: IBM HR Analytics Employee Attrition Dataset
- Size: 1,470 records
- Target variable: Attrition (Yes / No)

## Requirements
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Project Structure
1. Data Loading
2. Data Quality Checks
3. Preprocessing & Feature Engineering
4. Exploratory Data Analysis**
5. Model Training
6. Model Evaluation
7. Conclusion

## Machine Learning Models
### Logistic Regression
- Used for interpretability
- Handles linear relationships well

### Decision Tree Classifier
- Captures non-linear patterns
- Used for feature importance analysis

## Key Findings
- Attrition is influenced by:
  - Overtime -> Employees who stay overtime are more likely to leave the company
  - Age -> People younger than 21 are the most likely to leave the company. The Lowest attrition is in the ages 35-55.
  - Monthly income -> Average income of people who leave is less than those who stay
  - Job and work-life satisfaction -> Especially job satisfaction and environment satisfaction are negatively correlated with attrition
    
- Logistic Regression achieved a higher overall accuracy
