# fairml

**Bias Detection and Mitigation Toolkit for Responsible AI**

---

## 📝 **Overview**

`fairml` is a Python package designed to help data scientists and machine learning engineers **detect, visualize, and mitigate bias** in datasets and models seamlessly.

With intuitive APIs, rich visualizations, and practical mitigation techniques, `fairml` makes fairness-aware AI development accessible to every practitioner.

---

## ⚡ **Key Features**

✅ Bias detection metrics  
✅ Bias visualization dashboards  
✅ Pre-processing and post-processing mitigation methods  
✅ scikit-learn compatible pipelines  
✅ Lightweight, modular, and extensible design

---

## 💻 **Installation**

```
pip install fairml
```

*(Currently under development; install via GitHub for now)*

```
pip install git+https://github.com/nik21hil/fairml.git
```

---

## 🚀 **Quickstart**

```python
from fairml import detection

# Example: Calculate Statistical Parity Difference
spd = detection.statistical_parity_difference(y_pred, sensitive_features, privileged_group='A', unprivileged_group='B')
print("Statistical Parity Difference:", spd)
```

More detailed examples and notebooks are available in the `examples/` folder.

---

## 📊 **Modules**

- `detection.py`: Bias detection metrics  
- `visualization.py`: Fairness visualizations and plots  
- `mitigation.py`: Bias mitigation algorithms  
- `utils.py`: Helper functions

---

## 🔧 **Fairness Mitigation Utilities**

The `mitigation` module includes several techniques to reduce class imbalance and improve fairness.

### ✅ Supported Methods

| Method                       | Type           | Description                                                                 |
|-----------------------------|----------------|-----------------------------------------------------------------------------|
| `apply_smote`               | Over-sampling  | Generates synthetic minority samples using k-nearest neighbors             |
| `apply_adasyn`              | Over-sampling  | Focuses more on hard-to-learn minority samples                             |
| `combined_resample`         | Combo          | Combines `SMOTE` with `Tomek Links` for noise reduction                    |
| `apply_cluster_centroids`   | Under-sampling | Replaces majority class samples with cluster centroids using KMeans       |

---

## 🧪 **Usage Examples**

```python
from fairml.mitigation import (
    apply_smote,
    apply_adasyn,
    combined_resample,
    apply_cluster_centroids
)

# Sample imbalanced data
X = pd.DataFrame({'f1': range(30)})
y = np.array([0] * 25 + [1] * 5)

# SMOTE
X_sm, y_sm = apply_smote(X, y)

# ADASYN
X_ad, y_ad = apply_adasyn(X, y)

# SMOTE + Tomek Links
X_comb, y_comb = combined_resample(X, y, strategy='smote_tomek')

# Cluster Centroids
X_cc, y_cc = apply_cluster_centroids(X, y)
```

---

### 🛠 **When to Use What?**

| Scenario                                | Recommended Method     |
|----------------------------------------|------------------------|
| Moderate imbalance                     | `apply_smote`          |
| Imbalance + noisy boundaries           | `combined_resample`    |
| Focus on harder minority examples      | `apply_adasyn`         |
| Overwhelming majority class size       | `apply_cluster_centroids` |

---

## 📁 **Project Structure**

```
fairml/
│
├── fairml/
│   ├── detection.py
│   ├── visualization.py
│   ├── mitigation.py
│   ├── utils.py
│
├── examples/
├── tests/
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ✅ **Roadmap**

- [x] Phase 0: Project setup  
- [x] Phase 1: Core bias detection metrics  
- [x] Phase 2: Visualization module  
- [x] Phase 3: Pre-processing mitigation techniques  
- [x] Phase 4: Post-processing mitigation techniques  
- [ ] Phase 5: End-to-end example notebooks  
- [ ] Phase 6: PyPI release

---

## 🤝 **Contributing**

Contributions are welcome. Please open an issue or pull request to discuss improvements, features, or bug fixes.

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ **Acknowledgements**

- IBM AI Fairness 360  
- Microsoft Fairlearn  
- The broader AI fairness and ethics research community

---

## 🌐 **Author**

**Nikhil Singh**  
[GitHub](https://github.com/nik21hil) | [LinkedIn](https://www.linkedin.com/in/nikhil-singh21/)
