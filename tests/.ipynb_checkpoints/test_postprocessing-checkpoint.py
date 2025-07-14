import numpy as np
import pytest
from fairml.postprocessing import (
    reject_option_classification,
    equalized_odds_adjustment,
    threshold_optimization
)

def test_reject_option_classification():
    y_prob = np.array([0.2, 0.5, 0.8, 0.4, 0.6])
    y_pred = np.array([0, 1, 1, 0, 1])
    sensitive = np.array(['A', 'A', 'B', 'A', 'B'])
    adjusted = reject_option_classification(y_prob, y_pred, sensitive, group_a='A', group_b='B', low=0.3, high=0.7)
    expected = np.array([0, 1, 1, 1, 0])  # Index 2 flipped to 0 (B), Index 3 flipped to 1 (A)
    assert np.array_equal(adjusted, expected)

def test_equalized_odds_adjustment():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.6, 0.4])
    sensitive = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    adjusted_preds, thresholds = equalized_odds_adjustment(y_true, y_prob, sensitive)
    assert isinstance(adjusted_preds, np.ndarray)
    assert set(thresholds.keys()) == {'A', 'B'}
    assert all(0 <= t <= 1 for t in thresholds.values())

def test_threshold_optimization_tpr():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5])
    sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    thresholds = threshold_optimization(y_true, y_prob, sensitive, 'A', 'B', metric='tpr')
    assert 'A' in thresholds and 'B' in thresholds
    assert 0 < thresholds['A'] < 1
    assert 0 < thresholds['B'] < 1

def test_threshold_optimization_fpr():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5])
    sensitive = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    thresholds = threshold_optimization(y_true, y_prob, sensitive, 'A', 'B', metric='fpr')
    assert 'A' in thresholds and 'B' in thresholds
    assert 0 < thresholds['A'] < 1
    assert 0 < thresholds['B'] < 1


