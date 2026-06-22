# Customer Churn Prediction

A machine learning pipeline for predicting customer churn, developed as a senior capstone project (INDR 491) at Koç University.

## Problem
Identify customers at risk of churning using an anonymized dataset with 47 features — 20 categorical and 27 numerical variables — with a binary churn/no-churn target.

## Pipeline
1. **Exploratory Data Analysis** — class balance, distribution plots, correlation heatmaps per feature
2. **Missing Value Imputation** — MICE method via scikit-learn's `IterativeImputer`
3. **Feature Encoding** — One-hot encoding for all categorical variables
4. **Modeling**
   - Logistic Regression (baseline)
   - Gradient Boosting Classifier with `GridSearchCV` + `StratifiedKFold` (5-fold CV)
5. **Evaluation** — AUC-ROC on a labeled held-out test set

## Tech Stack
- Python 3
- scikit-learn · pandas · numpy · matplotlib · seaborn

## Notebooks
| Notebook | Description |
|---|---|
| `01_eda_preliminaries.ipynb` | EDA and descriptive statistics |
| `02_training_data_analysis.ipynb` | Training set exploration with missing values |
| `03_imputed_training_analysis.ipynb` | Post-imputation training data analysis |
| `04_test_data_analysis.ipynb` | Test set exploration with missing values |
| `05_imputed_test_analysis.ipynb` | Post-imputation test data analysis |
| `06_modeling_logistic_regression_gradient_boosting.ipynb` | Model training, hyperparameter tuning, and evaluation |

## Results
Gradient Boosting with tuned hyperparameters (grid-searched over `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) outperformed the logistic regression baseline on AUC-ROC.
