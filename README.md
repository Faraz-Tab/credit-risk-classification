# Credit Risk Classification

Logistic regression model that flags high-risk loans in a heavily imbalanced lending dataset (30:1 healthy to high-risk).

## Overview

A missed high-risk loan costs a lender far more than a healthy loan sent for extra review, so the model is evaluated primarily on **recall for the high-risk class**, with balanced accuracy as the overall metric. A baseline model is compared against the same model trained on oversampled data.

## Data

`Credit_Risk/Resources/lending_data.csv`: 77,536 loans, 2,500 (3.2%) labelled high-risk.

Features: loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt. Target: `loan_status` (0 = healthy, 1 = high-risk).

The dataset was provided as teaching data in the University of Toronto SCS Data Analytics Certificate.

## Approach

1. Stratified 75/25 train/test split, so both sets keep the 3.2% high-risk share.
2. Pipeline: `StandardScaler` → (optional) `RandomOverSampler` → `LogisticRegression`. Scaling is required for the solver to converge on these features.
3. **Baseline:** trained on the original class balance.
4. **Oversampled:** high-risk loans in the training set only are resampled to a 0.18 ratio; the test set is untouched.
5. Evaluation: balanced accuracy, confusion matrix, per-class precision and recall.

## Results

Test set: 19,384 loans, 625 high-risk. Reproduced with the pinned versions in `requirements.txt`.

| Model | Balanced accuracy | High-risk recall | High-risk precision | High-risk missed | False alarms |
|---|---|---|---|---|---|
| Baseline | 0.986 | 0.978 | 0.872 | 14 | 90 |
| Oversampled (0.18) | **0.994** | **0.992** | 0.872 | **5** | 91 |

Oversampling cut missed high-risk loans from 14 to 5 for one additional false alarm, with no change in precision. For a lender, that is the better trade-off.

Precision stays at 0.87 for both models: about 1 in 8 flagged loans is healthy. Oversampling duplicates existing high-risk rows rather than adding information, so it improves recall but cannot improve precision. More high-risk samples or additional features would be needed for that.

`Credit_Risk/MODEL_REPORT.md` is the written report from the original analysis, which used unscaled features and an unstratified split. Its figures and conclusions reflect that earlier run.

## Project Structure

```
├── Credit_Risk/
│   ├── credit_risk_classification.ipynb   # analysis narrative
│   ├── MODEL_REPORT.md                    # original written report
│   └── Resources/lending_data.csv
├── src/credit_risk.py                     # data loading, pipelines, evaluation
├── tests/test_credit_risk.py
├── requirements.txt
└── requirements-dev.txt
```

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
jupyter notebook Credit_Risk/credit_risk_classification.ipynb
```

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · pytest · GitHub Actions
