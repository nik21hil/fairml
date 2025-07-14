""
"""
Visualization utilities for inspecting bias metrics and mitigation impact.

Includes:
- 📈 plot_group_metrics(): Metric bar plots for each group
- 📊 plot_fairness_grid(): Accuracy vs. Fairness heatmap
- 🌀 plot_disparity_flow(): Sankey-style flow for before/after predictions
- 🎯 plot_threshold_impact(): Threshold sweep visualization
- 🧠 plot_metric_dashboard(): Multi-metric compact dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from fairml.detection import (
    statistical_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
    true_positive_rate_difference,
    false_positive_rate_difference,
    accuracy_difference
)

def plot_group_metrics(y_true, y_pred, sensitive_features, group_names=None):
    metrics = {
        'Accuracy': lambda yt, yp: np.mean(yt == yp),
        'TPR': lambda yt, yp: np.sum((yt == 1) & (yp == 1)) / np.sum(yt == 1),
        'FPR': lambda yt, yp: np.sum((yt == 0) & (yp == 1)) / np.sum(yt == 0),
        'Precision': lambda yt, yp: np.sum((yt == 1) & (yp == 1)) / np.sum(yp == 1) if np.sum(yp == 1) > 0 else 0,
    }

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': sensitive_features})
    results = []

    for g in df['group'].unique():
        subset = df[df['group'] == g]
        for name, func in metrics.items():
            try:
                val = func(subset['y_true'], subset['y_pred'])
            except:
                val = np.nan
            results.append({'Group': g, 'Metric': name, 'Value': val})

    plot_df = pd.DataFrame(results)
    if group_names:
        group_map = dict(zip(df['group'].unique(), group_names))
        plot_df['Group'] = plot_df['Group'].map(group_map)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Metric', y='Value', hue='Group')
    plt.title("Group-wise Fairness Metrics")
    plt.ylim(0, 1)
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

def plot_fairness_grid(results_df, fairness_metric="SPD", accuracy_col="accuracy"):
    pivot_df = results_df.pivot(index="method", columns=fairness_metric, values=accuracy_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Accuracy"})
    plt.title(f"Fairness vs Accuracy Grid ({fairness_metric})")
    plt.ylabel("Mitigation Method")
    plt.xlabel(fairness_metric)
    plt.tight_layout()
    plt.show()

def plot_disparity_flow(y_true_before, y_pred_before, y_true_after, y_pred_after, sensitive_features):
    df = pd.DataFrame({
        'group': sensitive_features,
        'before': y_pred_before,
        'after': y_pred_after
    })
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = df['group'].unique()
    widths = [df[df['group'] == g]['before'].sum() for g in groups]
    new_widths = [df[df['group'] == g]['after'].sum() for g in groups]

    for i, g in enumerate(groups):
        ax.barh(i, widths[i], color='skyblue', height=0.4, label='Before' if i == 0 else "")
        ax.barh(i + 0.5, new_widths[i], color='salmon', height=0.4, label='After' if i == 0 else "")
        ax.text(widths[i] + 0.5, i, f"{widths[i]}", va='center')
        ax.text(new_widths[i] + 0.5, i + 0.5, f"{new_widths[i]}", va='center')

    ax.set_yticks(np.arange(len(groups)) + 0.25)
    ax.set_yticklabels(groups)
    ax.set_title("Disparity Flow Before vs After Mitigation")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_threshold_impact(thresholds, metric_values, metric_name="Statistical Parity"):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, metric_values, marker='o')
    plt.title(f"Threshold Impact on {metric_name}")
    plt.xlabel("Threshold")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metric_dashboard(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group):
    spd = statistical_parity_difference(y_pred, sensitive_features, privileged_group, unprivileged_group)
    di = disparate_impact(y_pred, sensitive_features, privileged_group, unprivileged_group)
    tpr_diff = equal_opportunity_difference(y_true, y_pred, sensitive_features, privileged_group,
                                            unprivileged_group)
    fpr_diff = false_positive_rate_difference(y_true, y_pred, sensitive_features, privileged_group,
                                            unprivileged_group)
    acc_diff = accuracy_difference(y_true, y_pred, sensitive_features, privileged_group,
                                            unprivileged_group)

    metrics = {
        "Statistical Parity": spd,
        "Disparate Impact": di,
        "TPR Difference": tpr_diff,
        "FPR Difference": fpr_diff,
        "Accuracy Diff": acc_diff
    }

    plt.figure(figsize=(12, 5))
    for i, (name, value) in enumerate(metrics.items()):
        plt.subplot(1, 5, i + 1)
        plt.bar([name], [value], color='teal')
        plt.ylim(-1, 1)
        plt.xticks(rotation=45, ha='right')
        plt.title(name)
        plt.axhline(0, color='gray', linestyle='--')
    plt.suptitle("Fairness Metric Dashboard")
    plt.tight_layout()
    plt.show()



