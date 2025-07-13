import pytest
from fairml.visualization import (
    plot_group_metric_bar,
    plot_fairness_tradeoff,
    plot_pre_post_comparison
)

def test_plot_group_metric_bar():
    group_names = ['Male', 'Female']
    metric_values = [0.8, 0.6]
    try:
        plot_group_metric_bar(group_names, metric_values, metric_name='TPR')
    except Exception as e:
        pytest.fail(f"plot_group_metric_bar raised an exception: {e}")

def test_plot_fairness_tradeoff():
    fairness_scores = [0.1, 0.3, 0.5]
    accuracy_scores = [0.92, 0.89, 0.86]
    labels = ['Base', 'Weighting', 'Resampling']
    try:
        plot_fairness_tradeoff(fairness_scores, accuracy_scores, labels)
    except Exception as e:
        pytest.fail(f"plot_fairness_tradeoff raised an exception: {e}")

def test_plot_pre_post_comparison():
    metrics_before = [0.4, 0.6]
    metrics_after = [0.2, 0.3]
    metric_names = ['SPD', 'DIR']
    try:
        plot_pre_post_comparison(metrics_before, metrics_after, metric_names)
    except Exception as e:
        pytest.fail(f"plot_pre_post_comparison raised an exception: {e}")


