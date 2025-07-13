import numpy as np
import pandas as pd

def reweight_samples(y, sensitive_features, privileged_group, unprivileged_group):
    """
    Computes sample weights to balance the distribution of outcomes across groups.

    Parameters:
    - y: array-like, true labels (binary)
    - sensitive_features: array-like, sensitive attribute values
    - privileged_group: value indicating privileged group
    - unprivileged_group: value indicating unprivileged group

    Returns:
    - sample_weights: np.array of weights (same length as y)
    """
    y = np.array(y)
    sensitive_features = np.array(sensitive_features)

    sample_weights = np.ones_like(y, dtype=float)

    # Group sizes
    total = len(y)
    group_counts = {
        privileged_group: np.sum(sensitive_features == privileged_group),
        unprivileged_group: np.sum(sensitive_features == unprivileged_group)
    }

    for group in [privileged_group, unprivileged_group]:
        group_mask = (sensitive_features == group)
        group_weight = total / group_counts[group]
        sample_weights[group_mask] = group_weight

    return sample_weights


def resample_dataset(X, y, sensitive_features, privileged_group, unprivileged_group, strategy='undersample'):
    """
    Resamples dataset to balance groups.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: array-like, labels
    - sensitive_features: array-like, sensitive attribute values
    - privileged_group: value for privileged group
    - unprivileged_group: value for unprivileged group
    - strategy: 'undersample' or 'oversample'

    Returns:
    - X_resampled, y_resampled, sensitive_features_resampled
    """
    df = X.copy()
    df['label'] = y
    df['group'] = sensitive_features

    df_priv = df[df['group'] == privileged_group]
    df_unpriv = df[df['group'] == unprivileged_group]

    if strategy == 'undersample':
        min_size = min(len(df_priv), len(df_unpriv))
        df_priv = df_priv.sample(min_size, random_state=42)
        df_unpriv = df_unpriv.sample(min_size, random_state=42)
    elif strategy == 'oversample':
        max_size = max(len(df_priv), len(df_unpriv))
        df_priv = df_priv.sample(max_size, replace=True, random_state=42)
        df_unpriv = df_unpriv.sample(max_size, replace=True, random_state=42)
    else:
        raise ValueError("strategy must be 'undersample' or 'oversample'")

    df_balanced = pd.concat([df_priv, df_unpriv]).sample(frac=1.0, random_state=42)

    return (
        df_balanced.drop(columns=['label', 'group']),
        df_balanced['label'].values,
        df_balanced['group'].values
    )



