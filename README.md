# Customer Churn Prediction in Telecommunications Using Machine Learning

This project aims to predict customer churn in the telecommunications industry using machine learning, focusing on improving customer retention strategies. The project employs classification models such as XGBoost with Optuna tuning, and provides a complete end-to-end pipeline including preprocessing, training, evaluation, and visualization using Power BI.

## Problem Overview

In the telecommunications industry, customer churn—when a customer stops using a service—is a major challenge. Identifying customers who are likely to churn can help companies take proactive measures to retain them. However, manually detecting churn patterns is difficult and inefficient. A machine learning-based solution provides a scalable and data-driven approach to tackle this problem.

## Objectives

- To build a robust machine learning model that accurately predicts whether a customer will churn.
- To evaluate model performance using metrics such as accuracy, F1-score, and precision.
- To visualize insights and patterns through Power BI to support decision-making.

## Workflow Overview

The project follows a structured workflow:

1. **Data Loading & Preprocessing**
   - Dataset: `Telco-Customer-Churn.csv`
   - Convert `TotalCharges` to numeric
   - Remove rows with missing values
   - Encode categorical features using `LabelEncoder`
   - Drop non-informative ID columns

2. **Model Training**
   - Use `XGBoostClassifier` as the primary model
   - Hyperparameter optimization using `Optuna` and Stratified K-Fold (5-fold)
   - Save the best model from the best-performing fold

3. **Evaluation**
   - Metrics: Accuracy, F1 Score, Precision (macro)
   - Confusion Matrix visualization
   - Feature importance analysis from trained model

4. **Power BI Visualization**
   - Integrate model and predictions into Power BI
   - Display feature importance, churn distribution, correlation matrix, and churn segmentation

## Visualizations (Power BI)

The following visualizations are included:

- **Customer Churn Distribution**  

  <img src="./Customer Churn Distribution.png" alt="Customer Churn Distribution" width="300"/>

  This bar chart shows the distribution of customer churn in the dataset. The number of customers who did not churn is significantly higher (5.2K) compared to those who churned (1.9K). 
  

- **Feature Importance Based on XGBoost Model**  
  <img src="./Feature Importance Based on XGBoost Model.png" alt="Feature Importance Based on XGBoost Model" width="400"/>

  This bar chart displays the relative importance of each feature used in the XGBoost model for predicting customer churn. The `Contract` type stands out as the most influential feature, contributing approximately 41% to the model's decisions. Other relevant features include `TechSupport`, `OnlineSecurity`, and `InternetService`, while features like `Partner` and `gender` have minimal impact.


- **Churn Rate by Contract Type**  
  <img src="./Churn Rate by Contract Type.png" alt="Churn by Contract Type" width="150"/>
  
  This 100% stacked column chart visualizes the proportion of churn (Yes/No) across different contract types. Customers with month-to-month contracts have the highest churn rate at 42.71%, indicating they are more likely to leave the service. In contrast, churn is significantly lower among customers with one-year and two-year contracts, suggesting longer commitments reduce churn likelihood.


- **Correlation Between Monthly Charges, Total Charges, and Tenure**  
  <img src="./Correlation Between Monthly Charges, Total Charges, and Tenure.png" alt="Correlation Between Monthly Charges, Total Charges, and Tenure" width="300"/>

  This heatmap visualizes the Pearson correlation between three numerical features: `MonthlyCharges`, `TotalCharges`, and `tenure`.  The strongest positive correlation is observed between `TotalCharges` and `tenure` (0.83), suggesting that longer-tenured customers tend to accumulate higher charges.  Meanwhile, `MonthlyCharges` is moderately correlated with `TotalCharges` (0.65), but only weakly with `tenure` (0.25), indicating that monthly rates vary independently of how long a customer has stayed.



## Dataset

- Source: Telco Customer Churn Dataset (IBM Sample)
- Shape:  7043 rows × 21 columns
- Target column: `Churn` (`Yes` / `No`)

## Requirements

- Python 3.9+
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - matplotlib, seaborn
  - xgboost
  - scikit-learn
  - optuna
- Power BI Desktop (for visualization)

## Dataset Source

This project utilizes the **Telco Customer Churn** dataset, which was originally published for public use on Kaggle.

**Citation:**

> IBM Sample Data. *Telco Customer Churn*. Retrieved from Kaggle:  
> [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> Data files © Original Authors.


