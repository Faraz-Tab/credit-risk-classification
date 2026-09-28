import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit_risk import build_model, evaluate, load_data, split


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    return split(X, y)


def test_split_is_stratified(data):
    _, _, y_train, y_test = data
    assert abs(y_train.mean() - y_test.mean()) < 0.001


def test_oversampling_only_affects_training(data):
    X_train, X_test, y_train, y_test = data
    model = build_model(oversample_ratio=0.18).fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(y_test)


@pytest.mark.parametrize("ratio, min_bal_acc", [(None, 0.98), (0.18, 0.99)])
def test_model_performance(data, ratio, min_bal_acc):
    X_train, X_test, y_train, y_test = data
    results = evaluate(build_model(ratio).fit(X_train, y_train), X_test, y_test)
    assert results["balanced_accuracy"] >= min_bal_acc
