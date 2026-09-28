"""Data loading, model pipelines and evaluation for the credit risk classifier."""
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "Credit_Risk" / "Resources" / "lending_data.csv"
TARGET = "loan_status"
RANDOM_STATE = 1


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def split(X, y, test_size=0.25):
    # Stratify so the rare high-risk class keeps the same share in train and test
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def build_model(oversample_ratio=None):
    """Scaled logistic regression; optionally oversample the minority class on training data only."""
    steps = [("scale", StandardScaler())]
    if oversample_ratio is not None:
        steps.append(("oversample", RandomOverSampler(sampling_strategy=oversample_ratio, random_state=RANDOM_STATE)))
    steps.append(("clf", LogisticRegression(random_state=RANDOM_STATE)))
    return Pipeline(steps)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, target_names=["healthy", "high-risk"], digits=3),
    }
