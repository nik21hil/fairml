import numpy as np
import pandas as pd
import pytest
import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairml import visualization


@pytest.fixture
def dummy_data():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    sensitive = np.array(['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'])
    return y_true, y_pred, sensitive


def test_plot_group_metrics(dummy_data):
    y_true, y_pred, sensitive = dummy_data
    visualization.plot_group_metrics(y_true, y_pred, sensitive)
    plt.close()


def test_plot_fairness_grid():
    df = pd.DataFrame({
        'method': ['SMOTE', 'ADASYN', 'Reweighting'],
        'SPD': [0.1, -0.2, 0.0],
        'accuracy': [0.85, 0.82, 0.88]
    })
    visualization.plot_fairness_grid(df, fairness_metric='SPD', accuracy_col='accuracy')
    plt.close()


def test_plot_disparity_flow():
    y_true_b = np.array([1, 0, 1, 0])
    y_pred_b = np.array([1, 0, 1, 0])
    y_true_a = np.array([1, 0, 1, 0])
    y_pred_a = np.array([0, 1, 1, 0])
    sensitive = np.array(['A', 'A', 'B', 'B'])
    visualization.plot_disparity_flow(y_true_b, y_pred_b, y_true_a, y_pred_a, sensitive)
    plt.close()


def test_plot_threshold_impact():
    thresholds = np.linspace(0.1, 0.9, 9)
    spd_values = np.random.uniform(-0.2, 0.2, size=9)
    visualization.plot_threshold_impact(thresholds, spd_values, metric_name="SPD")
    plt.close()


def test_plot_metric_dashboard(dummy_data):
    y_true, y_pred, sensitive = dummy_data
    visualization.plot_metric_dashboard(y_true, y_pred, sensitive, privileged_group='A',
                                        unprivileged_group='B')
    plt.close()



