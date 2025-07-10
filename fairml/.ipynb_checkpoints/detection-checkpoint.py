import numpy as np
import pandas as pd

def statistical_parity_difference(y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Calculates Statistical Parity Difference (SPD).

    Parameters:
    - y_pred: array-like, model predictions (binary 0/1)
    - sensitive_features: array-like, sensitive attribute values per sample
    - privileged_group: value indicating privileged group in sensitive_features
    - unprivileged_group: value indicating unprivileged group in sensitive_features

    Returns:
    - SPD value (float)
    """
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)
    
    p_privileged = y_pred[sensitive_features == privileged_group].mean()
    p_unprivileged = y_pred[sensitive_features == unprivileged_group].mean()
    
    return p_unprivileged - p_privileged


def disparate_impact_ratio(y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Calculates Disparate Impact Ratio (DIR).

    Parameters:
    - y_pred: array-like, model predictions (binary 0/1)
    - sensitive_features: array-like, sensitive attribute values per sample
    - privileged_group: value indicating privileged group in sensitive_features
    - unprivileged_group: value indicating unprivileged group in sensitive_features

    Returns:
    - DIR value (float)
    """
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)
    
    p_privileged = y_pred[sensitive_features == privileged_group].mean()
    p_unprivileged = y_pred[sensitive_features == unprivileged_group].mean()
    
    if p_privileged == 0:
        return np.inf
    else:
        return p_unprivileged / p_privileged


def equal_opportunity_difference(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Calculates Equal Opportunity Difference (EOD).

    Parameters:
    - y_true: array-like, true binary labels
    - y_pred: array-like, model predictions (binary 0/1)
    - sensitive_features: array-like, sensitive attribute values per sample
    - privileged_group: value indicating privileged group in sensitive_features
    - unprivileged_group: value indicating unprivileged group in sensitive_features

    Returns:
    - EOD value (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)
    
    # True positive rates
    tpr_privileged = ((y_pred == 1) & (y_true == 1) & (sensitive_features == privileged_group)).sum() / \
                     ((y_true == 1) & (sensitive_features == privileged_group)).sum()
    
    tpr_unprivileged = ((y_pred == 1) & (y_true == 1) & (sensitive_features == unprivileged_group)).sum() / \
                       ((y_true == 1) & (sensitive_features == unprivileged_group)).sum()
    
    return tpr_unprivileged - tpr_privileged



