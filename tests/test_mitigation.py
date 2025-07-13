import numpy as np
import pandas as pd
from fairml.mitigation import reweight_samples, resample_dataset, apply_smote, apply_adasyn, apply_hybrid_sampling, combined_resample, apply_cluster_centroids

def test_reweight_samples_balances_weights():
    y = np.array([1, 0, 1, 0, 1, 0])
    sensitive = np.array(['A', 'A', 'B', 'B', 'B', 'B'])

    weights = reweight_samples(y, sensitive, privileged_group='A', unprivileged_group='B')

    # There are 2 'A' and 4 'B' → expect A weights > B weights
    assert len(weights) == len(y)
    assert weights[sensitive == 'A'][0] > weights[sensitive == 'B'][0]

def test_resample_dataset_undersample():
    X = pd.DataFrame({'f1': [1, 2, 3, 4, 5, 6]})
    y = np.array([1, 0, 1, 0, 1, 0])
    sensitive = np.array(['A', 'A', 'B', 'B', 'B', 'B'])

    X_res, y_res, sens_res = resample_dataset(
        X, y, sensitive,
        privileged_group='A',
        unprivileged_group='B',
        strategy='undersample'
    )

    # Should be balanced: 2 A, 2 B
    unique, counts = np.unique(sens_res, return_counts=True)
    assert all(count == counts[0] for count in counts)

def test_resample_dataset_oversample():
    X = pd.DataFrame({'f1': [1, 2, 3]})
    y = np.array([1, 0, 1])
    sensitive = np.array(['A', 'A', 'B'])

    X_res, y_res, sens_res = resample_dataset(
        X, y, sensitive,
        privileged_group='A',
        unprivileged_group='B',
        strategy='oversample'
    )

    # Now: A has 2 samples, B has 1 → expect oversample to 2 each → 4 total
    assert len(X_res) == 4
    unique, counts = np.unique(sens_res, return_counts=True)
    assert all(count == counts[0] for count in counts)


def test_apply_smote_balances_classes():
    X = pd.DataFrame({'f1': list(range(20))})
    y = np.array([0]*15 + [1]*5)  # Class imbalance
    X_res, y_res = apply_smote(X, y, k_neighbors=3)
    unique, counts = np.unique(y_res, return_counts=True)
    assert len(set(counts)) == 1

def test_apply_adasyn_balances_classes():
    X = pd.DataFrame({'f1': list(range(20))})
    y = np.array([0]*15 + [1]*5)
    X_res, y_res = apply_adasyn(X, y, n_neighbors=3)
    unique, counts = np.unique(y_res, return_counts=True)
    assert abs(counts[0] - counts[1]) <= 2

def test_apply_hybrid_sampling_balances_classes():
    X = pd.DataFrame({'f1': list(range(30))})
    y = np.array([0]*25 + [1]*5)  # Class imbalance

    X_res, y_res = apply_hybrid_sampling(X, y, smote_k_neighbors=3)
    unique, counts = np.unique(y_res, return_counts=True)

    # Check if classes are balanced or nearly balanced
    assert abs(counts[0] - counts[1]) <= 1

def test_combined_resample_smote_tomek_balances_classes():
    X = pd.DataFrame({'f1': list(range(30))})
    y = np.array([0]*25 + [1]*5)
    X_res, y_res = combined_resample(X, y, strategy='smote_tomek')
    
    # Check class balance roughly (not exact due to Tomek removal)
    from collections import Counter
    counts = Counter(y_res)
    assert abs(counts[0] - counts[1]) <= 5  # should be close to balanced

def test_combined_resample_invalid_strategy_raises_error():
    X = pd.DataFrame({'f1': list(range(10))})
    y = np.array([0]*8 + [1]*2)
    try:
        combined_resample(X, y, strategy='invalid')
    except ValueError as e:
        assert "Invalid strategy" in str(e)

def test_apply_cluster_centroids_balances_classes():
    X = pd.DataFrame({'f1': list(range(30))})
    y = np.array([0]*25 + [1]*5)  # Majority class: 0

    X_res, y_res = apply_cluster_centroids(X, y)
    count_0 = sum(y_res == 0)
    count_1 = sum(y_res == 1)
    assert count_0 == count_1












