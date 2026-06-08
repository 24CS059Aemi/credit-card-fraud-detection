"""Unit tests for data preprocessing."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


def make_dummy_df(n=1000, fraud_rate=0.02):
    np.random.seed(42)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount"] = np.abs(np.random.randn(n) * 100)
    data["Time"]   = np.arange(n, dtype=float)
    data["Class"]  = [1] * n_fraud + [0] * n_legit
    return pd.DataFrame(data)


def test_dataframe_shape():
    df = make_dummy_df()
    assert df.shape == (1000, 31)
    assert "Class" in df.columns


def test_fraud_rate():
    df = make_dummy_df(n=10000, fraud_rate=0.02)
    actual = df["Class"].mean()
    assert abs(actual - 0.02) < 0.005


def test_no_missing_values():
    df = make_dummy_df()
    assert df.isnull().sum().sum() == 0


def test_amount_non_negative():
    df = make_dummy_df()
    assert (df["Amount"] >= 0).all()


def test_class_binary():
    df = make_dummy_df()
    assert set(df["Class"].unique()).issubset({0, 1})
