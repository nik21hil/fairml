import matplotlib.pyplot as plt
import seaborn as sns

def plot_group_metric_bar(group_names, metric_values, metric_name='TPR', title=None):
    """
    Bar plot of a metric across sensitive groups.

    Parameters:
    - group_names: list of group labels (e.g., ['Male', 'Female'])
    - metric_values: list of metric values corresponding to each group
    - metric_name: name of the metric (e.g. 'TPR', 'FPR', 'Pos. Rate')
    - title: optional chart title
    """
    sns.barplot(x=group_names, y=metric_values)
    plt.ylabel(metric_name)
    plt.title(title or f'{metric_name} across Groups')
    plt.ylim(0, 1)
    plt.show()


def plot_fairness_tradeoff(fairness_scores, accuracy_scores, labels=None, title='Fairness vs Accuracy'):
    """
    Line plot showing fairness-accuracy trade-off.

    Parameters:
    - fairness_scores: list of fairness metric values
    - accuracy_scores: list of accuracy values
    - labels: optional labels for points (e.g., ['Baseline', 'After Mitigation'])
    - title: chart title
    """
    plt.figure(figsize=(6, 4))
    plt.plot(fairness_scores, accuracy_scores, marker='o')
    
    if labels:
        for i, label in enumerate(labels):
            plt.text(fairness_scores[i], accuracy_scores[i], label, fontsize=9)

    plt.xlabel('Fairness Score')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_pre_post_comparison(metrics_before, metrics_after, metric_names, title='Bias Metrics: Pre vs Post'):
    """
    Bar chart comparing pre- and post-mitigation metrics.

    Parameters:
    - metrics_before: list of metric values before mitigation
    - metrics_after: list of metric values after mitigation
    - metric_names: list of metric names (e.g. ['SPD', 'DIR'])
    - title: plot title
    """
    x = range(len(metric_names))
    width = 0.35

    plt.bar(x, metrics_before, width, label='Before')
    plt.bar([i + width for i in x], metrics_after, width, label='After')

    plt.xticks([i + width / 2 for i in x], metric_names)
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



