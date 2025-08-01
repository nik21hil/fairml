import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek, SMOTEENN


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


def apply_smote(X, y, k_neighbors=5):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def apply_adasyn(X, y, n_neighbors=5):
    ad = ADASYN(n_neighbors=n_neighbors, random_state=42)
    X_res, y_res = ad.fit_resample(X, y)
    return X_res, y_res

def apply_hybrid_sampling(X, y, smote_k_neighbors=5, under_sampling_strategy='auto'):
    """
    Apply hybrid sampling (SMOTE + Random UnderSampling) to balance the dataset.
    
    Parameters:
    - X: Features (DataFrame)
    - y: Labels (array-like)
    - smote_k_neighbors: Number of neighbors for SMOTE (default=5)
    - under_sampling_strategy: Strategy for under-sampling majority class (default='auto')
    
    Returns:
    - X_resampled, y_resampled: Resampled features and labels
    """
    over = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)
    under = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=42)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_res, y_res = pipeline.fit_resample(X, y)
    return X_res, y_res


def combined_resample(X, y, strategy='smote_tomek', random_state=42):
    """
    Apply combined resampling techniques to address class imbalance.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.array): Target array.
        strategy (str): Strategy to use - 'smote_tomek' or 'smote_enn'.
        random_state (int): Random state for reproducibility.

    Returns:
        X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    if strategy == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state, smote=smote)
    elif strategy == 'smote_enn':
        sampler = SMOTEENN(random_state=random_state, smote=smote)
    else:
        raise ValueError("Invalid strategy. Choose 'smote_tomek' or 'smote_enn'.")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

def apply_cluster_centroids(X, y, random_state=42):
    """
    Apply ClusterCentroids under-sampling to balance the dataset.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.array): Target array.
        random_state (int): Random seed for reproducibility.

    Returns:
        (X_resampled, y_resampled): Resampled dataset
    """
    cc = ClusterCentroids(random_state=random_state)
    X_res, y_res = cc.fit_resample(X, y)
    return X_res, y_res














