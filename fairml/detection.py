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


def disparate_impact(y_pred, sensitive_features, privileged_group, unprivileged_group):
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

def true_positive_rate_difference(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Difference in True Positive Rate (TPR) between groups.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)

    tpr_priv = ((y_pred == 1) & (y_true == 1) & (sensitive_features == privileged_group)).sum() / \
               ((y_true == 1) & (sensitive_features == privileged_group)).sum()

    tpr_unpriv = ((y_pred == 1) & (y_true == 1) & (sensitive_features == unprivileged_group)).sum() / \
                 ((y_true == 1) & (sensitive_features == unprivileged_group)).sum()

    return tpr_unpriv - tpr_priv


def false_positive_rate_difference(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Difference in False Positive Rate (FPR) between groups.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)

    fpr_priv = ((y_pred == 1) & (y_true == 0) & (sensitive_features == privileged_group)).sum() / \
               ((y_true == 0) & (sensitive_features == privileged_group)).sum()

    fpr_unpriv = ((y_pred == 1) & (y_true == 0) & (sensitive_features == unprivileged_group)).sum() / \
                 ((y_true == 0) & (sensitive_features == unprivileged_group)).sum()

    return fpr_unpriv - fpr_priv


def accuracy_difference(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group):
    """
    Difference in accuracy between groups.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)

    acc_priv = (y_true[sensitive_features == privileged_group] == y_pred[sensitive_features == privileged_group]).mean()
    acc_unpriv = (y_true[sensitive_features == unprivileged_group] == y_pred[sensitive_features == unprivileged_group]).mean()

    return acc_unpriv - acc_priv






