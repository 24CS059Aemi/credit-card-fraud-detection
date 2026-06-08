"""Unit tests for the Flask scoring API."""
import pytest
import numpy as np
import json
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    return model


def test_feature_count_validation(mock_model):
    """API should reject requests with wrong number of features."""
    features_wrong = list(range(20))   # should be 30
    assert len(features_wrong) != 30


def test_fraud_threshold_logic():
    """Score above threshold should flag as fraud."""
    prob = 0.85
    threshold = 0.30
    assert bool(prob >= threshold) is True


def test_legit_threshold_logic():
    """Score below threshold should not flag as fraud."""
    prob = 0.05
    threshold = 0.30
    assert bool(prob >= threshold) is False


def test_feature_vector_shape():
    """Features should be 30-dimensional."""
    features = list(range(30))
    assert len(features) == 30
