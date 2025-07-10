import numpy as np
from detection import statistical_parity_difference, disparate_impact_ratio, equal_opportunity_difference

# Dummy data
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
sensitive_features = np.array(['M', 'F', 'M', 'F', 'F', 'M', 'F', 'M'])

privileged_group = 'M'
unprivileged_group = 'F'

# Test SPD
spd = statistical_parity_difference(y_pred, sensitive_features, privileged_group, unprivileged_group)
print("Statistical Parity Difference:", spd)

# Test DIR
dir = disparate_impact_ratio(y_pred, sensitive_features, privileged_group, unprivileged_group)
print("Disparate Impact Ratio:", dir)

# Test EOD
eod = equal_opportunity_difference(y_true, y_pred, sensitive_features, privileged_group, unprivileged_group)
print("Equal Opportunity Difference:", eod)
