import numpy as np
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize


def reject_option_classification(y_prob, y_pred, sensitive_features, group_a, group_b, low=0.3, high=0.7):
    """
    Adjusts predictions near the decision boundary (between `low` and `high`) to reduce bias.
    Predictions for the unprivileged group are flipped to favorable outcome if uncertain.
    """
    y_adjusted = y_pred.copy()
    for i in range(len(y_prob)):
        if low <= y_prob[i] <= high:
            if sensitive_features[i] == group_a and y_pred[i] == 0:
                y_adjusted[i] = 1
            elif sensitive_features[i] == group_b and y_pred[i] == 1:
                y_adjusted[i] = 0
    return y_adjusted


def equalized_odds_adjustment(y_true, y_prob, sensitive_features):
    """
    Adjusts predicted probabilities to satisfy equalized odds constraint using ROC curves.
    """
    thresholds = {}
    adjusted_predictions = np.zeros_like(y_true)
    
    for group in np.unique(sensitive_features):
        group_idx = sensitive_features == group
        fpr, tpr, thresh = roc_curve(y_true[group_idx], y_prob[group_idx])
        # Use Youden’s J statistic to pick optimal threshold
        j_scores = tpr - fpr
        best_threshold = thresh[np.argmax(j_scores)]
        thresholds[group] = best_threshold
        adjusted_predictions[group_idx] = (y_prob[group_idx] >= best_threshold).astype(int)
    
    return adjusted_predictions, thresholds


def threshold_optimization(y_true, y_prob, sensitive_features, group_a, group_b, metric='tpr'):
    """
    Learns group-specific thresholds to equalize a specified fairness metric ('tpr' or 'fpr').
    """
    best_thresholds = {}
    thresholds = np.linspace(0.01, 0.99, 50)
    best_diff = float('inf')

    for t1 in thresholds:
        for t2 in thresholds:
            pred_a = y_prob[sensitive_features == group_a] >= t1
            pred_b = y_prob[sensitive_features == group_b] >= t2
            
            true_a = y_true[sensitive_features == group_a]
            true_b = y_true[sensitive_features == group_b]

            if metric == 'tpr':
                tpr_a = np.sum((pred_a == 1) & (true_a == 1)) / max(np.sum(true_a == 1), 1)
                tpr_b = np.sum((pred_b == 1) & (true_b == 1)) / max(np.sum(true_b == 1), 1)
                diff = abs(tpr_a - tpr_b)
            elif metric == 'fpr':
                fpr_a = np.sum((pred_a == 1) & (true_a == 0)) / max(np.sum(true_a == 0), 1)
                fpr_b = np.sum((pred_b == 1) & (true_b == 0)) / max(np.sum(true_b == 0), 1)
                diff = abs(fpr_a - fpr_b)
            else:
                raise ValueError("Unsupported metric. Choose 'tpr' or 'fpr'.")

            if diff < best_diff:
                best_diff = diff
                best_thresholds = {group_a: t1, group_b: t2}

    return best_thresholds


