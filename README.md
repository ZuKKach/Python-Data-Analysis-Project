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
4. Exploratory Data Analysis
5. Model Training
6. Model Evaluation
7. Conclusion

## Machine Learning Models
### Logistic Regression
- Used for interpretability
- Handles linear relationships well

### Decision Tree Classifier
- Captures non-linear patterns
- Feature importance analysis

## Key Findings
- Attrition is influenced by:
  - Overtime -> Employees who stay overtime are more likely to leave the company
  - Age -> People younger than 21 are the most likely to leave the company. The Lowest attrition is in the ages 35-55.
  - Monthly income -> Average income of people who leave is less than those who stay
  - Job and work-life satisfaction -> Especially job satisfaction and environment satisfaction are negatively correlated with attrition
- Logistic Regression achieved a higher overall accuracy



Charts:


<img width="640" height="480" alt="Figure_1 Employee Attrition Distribution" src="https://github.com/user-attachments/assets/ea5ca28a-e049-4be2-85d3-67d4c0c577fa" />
<img width="640" height="480" alt="Monthly Income vs Attrition" src="https://github.com/user-attachments/assets/85b52c5d-63a4-46be-8d4c-eeaa587ac192" />
<img width="640" height="480" alt="Age Distribution by Attrition" src="https://github.com/user-attachments/assets/e88c89d1-7f0f-4a9e-ab5e-481713fe3d5d" />
<img width="640" height="505" alt="Attrition by Department" src="https://github.com/user-attachments/assets/2c911ed0-7bad-45a0-af5c-aaebd7e936bb" />
<img width="640" height="480" alt="Attrition by Overtime" src="https://github.com/user-attachments/assets/e1d04793-a38b-48f9-861a-f21e8e8086fd" />
<img width="1200" height="600" alt="Happiness comparison" src="https://github.com/user-attachments/assets/e412baa4-c68b-4e36-a4cd-bcf62ced1644" />
<img width="1200" height="800" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/5cb0e3cf-492c-43b0-a115-c1c396ddf11d" />


    

