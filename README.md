# Fairml
> Bias Detection and Mitigation Toolkit for Responsible AI.
---

## ✨ **Overview**

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

```bash
pip install fairml
```

*(Currently under development; install via GitHub for now)*

```bash
pip install git+https://github.com/nik21hil/fairml.git
```

---

## 🚀 **Quickstart**

```python
from fairml import detection

# Example: Calculate Statistical Parity Difference
spd = detection.statistical_parity_difference(y_true, y_pred, sensitive_features)
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

## 🧰 **Mitigation Techniques**

The `fairml.mitigation` module offers multiple pre-processing techniques to handle class imbalance and fairness-aware reweighting:

### 🔄 Reweighting

```python
from fairml.mitigation import reweight_samples

weights = reweight_samples(y, sensitive_features, privileged_group='M', unprivileged_group='F')
```

### ⚖️ Resampling by Group

```python
from fairml.mitigation import resample_dataset

X_res, y_res, group_res = resample_dataset(X, y, sensitive_features,
                                           privileged_group='M',
                                           unprivileged_group='F',
                                           strategy='undersample')  # or 'oversample'
```

### 🔬 Synthetic Sampling Techniques

- **SMOTE**

```python
from fairml.mitigation import apply_smote

X_res, y_res = apply_smote(X, y)
```

- **ADASYN**

```python
from fairml.mitigation import apply_adasyn

X_res, y_res = apply_adasyn(X, y)
```

- **Hybrid Sampling (SMOTE + RandomUnderSampler)**

```python
from fairml.mitigation import apply_hybrid_sampling

X_res, y_res = apply_hybrid_sampling(X, y)
```

- **Combined Sampling**

```python
from fairml.mitigation import combined_resample

X_res, y_res = combined_resample(X, y, strategy='smote_tomek')  # or 'smote_enn'
```

- **Cluster Centroids (Under-sampling)**

```python
from fairml.mitigation import apply_cluster_centroids

X_res, y_res = apply_cluster_centroids(X, y)
```

These techniques help create a balanced dataset before training, reducing unfair bias in imbalanced classes.

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
- [ ] Phase 4: Post-processing mitigation techniques  
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
[GitHub](https://github.com/nik21hil) | [LinkedIn](https://www.linkedin.com/in/nkhlsngh/)

---

Enjoy building! 🎯
