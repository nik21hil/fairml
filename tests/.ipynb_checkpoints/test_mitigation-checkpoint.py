import numpy as np
import pandas as pd
from fairml.mitigation import reweight_samples, resample_dataset

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





