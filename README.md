# Corporate Bankruptcy Prediction with AI Assistant

An end-to-end Machine Learning project that predicts corporate bankruptcy risk from financial indicators, estimates financial losses, and provides an interactive Gradio demo for real-time analysis.

## Project Overview

This project addresses a real business problem: identifying companies at risk of bankruptcy using financial data. It combines:

- **Classification** (Random Forest) to predict whether a company is bankrupt or healthy
- **Regression** (Random Forest Regressor) to estimate the financial loss (Net Income to Total Assets)
- **An interactive Gradio web app** that lets users analyze a company and get an interpretation in natural language

## Dataset

- **Source:** `data.csv` — a financial dataset of companies with **96 features** (financial ratios) and 1 target column `Bankrupt?`
- **Target (classification):** `Bankrupt?` → `1` = bankrupt, `0` = healthy
- **Target (regression):** `Net Income to Total Assets`
- **Size:** ~6,800 rows, 96 columns
- Features include ROA, operating margins, debt ratios, cash flow rates, turnover ratios, etc.

## Methodology

### 1. Data Cleaning
- Converted `Bankrupt?` to numeric, dropped missing values
- Kept only valid labels `{0, 1}`
- Stripped whitespace from column names for consistency

### 2. Classification — Bankruptcy Prediction
- **Model:** `RandomForestClassifier` (100 estimators, `random_state=42`)
- **Split:** 80/20 train-test, **stratified** to handle class imbalance
- **Evaluation metrics:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **Feature importance:** top 10 most predictive financial ratios

### 3. Regression — Financial Loss Estimation
- **Model:** `RandomForestRegressor` (100 estimators, `random_state=42`)
- **Target:** `Net Income to Total Assets`
- **Evaluation metrics:** RMSE, R²
- **Visualizations:** predicted vs. actual, residual error plot

### 4. Interactive Demo (Gradio)
- A **Gradio Blocks** app that:
  - Takes a company sample from the test set
  - Returns the bankruptcy prediction (RISK / HEALTHY)
  - Returns the estimated financial loss
  - Generates a natural-language interpretation of the result

## Results

### Classification
| Metric | Class 0 (Healthy) | Class 1 (Bankrupt) |
|---|---|---|
| Precision | 0.97 | 0.73 |
| Recall | 1.00 | 0.18 |
| F1-score | 0.99 | 0.29 |

- **Overall accuracy:** 0.97
- **ROC-AUC:** ~0.90+
- The model is very strong at detecting healthy companies, but recall on bankrupt companies is low (class imbalance issue — a known challenge in bankruptcy prediction).

### Regression
- **RMSE:** 0.0024
- **R²:** 0.9964
- The regression model fits the data very well (R² ≈ 0.996).

## Technologies Used

- **Language:** Python 3
- **Data:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (RandomForestClassifier, RandomForestRegressor, train_test_split, metrics)
- **Visualization:** Matplotlib, Seaborn
- **Web app:** Gradio
- **Environment:** Google Colab

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ghaliahannachi2-rgb/bankruptcy-prediction.git
   cd bankruptcy-prediction
