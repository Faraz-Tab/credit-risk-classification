# Credit Risk Classification

Supervised machine learning model that identifies high-risk loans from lending data — a heavily imbalanced binary classification problem (30:1 class ratio).

## Overview

Lenders lose far more from a missed high-risk loan than from flagging a healthy one, so this project prioritizes **recall on the minority class**. A baseline logistic regression is compared against the same model trained on resampled data.

## Data

77,536 historical loans with features including loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, and derogatory marks. Only 2,500 loans (3.2%) are labelled high-risk.

## Approach

1. Train/test split with stratified evaluation
2. **Baseline:** logistic regression on the original imbalanced data
3. **Resampled:** `RandomOverSampler` (imbalanced-learn) applied to the training set, then logistic regression retrained
4. Evaluation via balanced accuracy, confusion matrix, and per-class precision/recall

## Results

| Model | Balanced Accuracy | High-Risk Recall | High-Risk Precision |
|---|---|---|---|
| Logistic Regression (original) | 0.952 | 0.91 | 0.85 |
| Logistic Regression (oversampled) | **0.991** | **0.99** | 0.84 |

Oversampling cut missed high-risk loans from 56 to 8 (out of 619 in the test set) at a negligible precision cost — the right trade-off for a lending context.

## Repository

- `Credit_Risk/credit_risk_classification.ipynb` — full analysis
- `Credit_Risk/report-template.md` — written model report
- `Credit_Risk/Resources/lending_data.csv` — dataset

**Tools:** Python · pandas · scikit-learn · imbalanced-learn
