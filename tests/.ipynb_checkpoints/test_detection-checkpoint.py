import numpy as np
from fairml.detection import statistical_parity_difference, disparate_impact_ratio, equal_opportunity_difference

def test_statistical_parity_difference():
    y_pred = np.array([1, 0, 1, 0])
    sensitive_features = np.array(['A', 'B', 'A', 'B'])
    result = statistical_parity_difference(y_pred, sensitive_features, privileged_group='A', unprivileged_group='B')
    assert np.isclose(result, -1.0)  # Adjust expected value based on calculation

def test_disparate_impact_ratio():
    y_pred = np.array([1, 0, 1, 0])
    sensitive_features = np.array(['A', 'B', 'A', 'B'])
    result = disparate_impact_ratio(y_pred, sensitive_features, privileged_group='A', unprivileged_group='B')
    assert np.isclose(result, 0.0)  # Adjust expected value based on calculation

def test_equal_opportunity_difference():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    sensitive_features = np.array(['A', 'B', 'A', 'B'])
    result = equal_opportunity_difference(y_true, y_pred, sensitive_features, privileged_group='A', unprivileged_group='B')
    assert np.isclose(result, -1.0)  # Adjust expected value based on calculation


