import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_pipeline import (
    load_data,
    clean_data,
    prepare_data,
    engineer_features,
    scale_features,
    train_model,
    evaluate_model,
)


def make_synthetic_df(rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "team": rng.integers(1, 10, size=rows),
        "targeted_productivity": rng.uniform(0.5, 0.9, size=rows),
        "smv": rng.uniform(10, 20, size=rows),
        "wip": rng.integers(500, 2000, size=rows),
        "over_time": rng.integers(0, 5000, size=rows),
        "incentive": rng.integers(0, 1000, size=rows),
        "idle_time": rng.integers(0, 500, size=rows),
        "idle_men": rng.integers(0, 10, size=rows),
        "no_of_style_change": rng.integers(0, 3, size=rows),
        "no_of_workers": rng.integers(20, 60, size=rows),
        "department": rng.choice(["sweing", "finishing"], size=rows),
        "quarter": rng.choice(["Quarter1", "Quarter2", "Quarter3", "Quarter4"], size=rows),
        "day": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], size=rows),
        "date": pd.date_range("2015-01-01", periods=rows, freq="D").astype(str),
        "actual_productivity": rng.uniform(0.4, 0.95, size=rows),
    })
    return df


def test_clean_and_prepare_split():
    df = make_synthetic_df(60)
    df_clean = clean_data(df, "actual_productivity")
    X_train, X_test, y_train, y_test = prepare_data(df_clean, "actual_productivity", test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df_clean)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_engineer_and_scale_features():
    df = make_synthetic_df(60)
    df_clean = clean_data(df, "actual_productivity")
    X_train, X_test, y_train, y_test = prepare_data(df_clean, "actual_productivity", test_size=0.25, random_state=42)
    X_train_eng, X_test_eng = engineer_features(X_train, X_test)
    # Check new features exist
    for col in ["total_time", "productive_time", "work_complexity", "worker_utilization", "incentive_per_worker", "wip_per_worker"]:
        assert col in X_train_eng.columns
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_eng, X_test_eng)
    assert X_train_scaled.shape == X_train_eng.shape
    assert X_test_scaled.shape == X_test_eng.shape


def test_train_and_evaluate_model():
    df = make_synthetic_df(80)
    df_clean = clean_data(df, "actual_productivity")
    X_train, X_test, y_train, y_test = prepare_data(df_clean, "actual_productivity", test_size=0.3, random_state=42)
    X_train_eng, X_test_eng = engineer_features(X_train, X_test)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_eng, X_test_eng)
    model = train_model(X_train_scaled, y_train, hyperparameter_tuning=False, random_state=42)
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, verbose=False)
    # Basic presence of metrics
    for key in ["train_r2", "test_r2", "train_rmse", "test_rmse", "train_mae", "test_mae", "test_mape", "accuracy_5pct", "accuracy_10pct", "r2_difference", "is_overfitting"]:
        assert key in metrics
