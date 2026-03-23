
---
# **Arbitrary Polynomial Chaos Expansion (aPCE) Framework**

This repository provides a modular and extensible MATLAB implementation of **Arbitrary Polynomial Chaos Expansion (aPCE)** for data-driven surrogate modeling, uncertainty quantification (UQ), and global sensitivity analysis (GSA).
The framework supports polynomial construction, regression, optional dimension reduction, automatic hyperparameter search, and unified prediction.

It is designed as a lightweight but research-grade aPCE engine for high-dimensional UQ problems.

---

## ✨ Features Overview

### **1. Data Processing**

* Standardization using sample mean μ and standard deviation σ
* Automatic removal of zero-variance dimensions
* Consistent transformation for both training and prediction
* Mapping to the standardized **U-space**

---

### **2. Polynomial Basis Construction**

Two formulations supported:

* **2012 version** (moment-based Hankel system)
* **2018 version** (generalized moment formulation)

Capabilities:

* Multi-index generation (supports q-norm truncation)
* Univariate orthogonal / orthonormal polynomial construction
* Tensor-product multivariate basis assembly Ψ

---

### **3. Regression Methods**

Four regression solvers included:

* **OLS**
* **LARS** (default)
* **Ridge**
* **Lasso**

All solvers output:

* PCE coefficients
* LOO / modified LOO
* Empirical error
* Active basis terms

Unified interface: `aPC_RegressionMethods`.

---

### **4. Optional Dimension Reduction (PLS / PCA)**

* Supports **PLS** or **PCA** after standardization
* `red_dim` can be a scalar or a vector
* `aPC_Train` automatically selects the dimension giving the **best LOO**
* Prediction applies the stored projection matrix

---

### **5. Automatic Hyperparameter Selection**

The following options support **scalar or vector input**:

* Polynomial order `Order`
* Truncation parameter `qnorm`
* Reduced dimension `red_dim`

`aPC_Train`:

1. Generates all parameter combinations
2. Trains models for each
3. Computes LOO
4. Selects the best model automatically

---

### **6. Prediction Pipeline**

Prediction automatically performs:

1. Standardization (using training μ, σ)
2. Dimension reduction (PLS/PCA)
3. Rebuilding univariate polynomials
4. Constructing Ψ
5. Computing final predictions

Interface: `aPC_Predict`.

---

### **7. Uncertainty Quantification (UQ)**

Provides both analytic and sample-based quantities:

* Mean
* Variance
* Standard deviation
* Skewness
* Kurtosis
* 95% confidence interval (CI)
* VaR95 / CVaR95

Interface: `aPC_printUQ`.

---

### **8. Sensitivity Analysis**

Computes Sobol’ indices:

* First-order
* Second-order
* Total effect

Interface: `aPC_ComputeSensitivity`.

---

### **9. Visualization**

Includes:

* UQ plots
* Sobol heatmaps
* PCE diagnostic plots

Located in the `display/` folder.

---

## 📁 Directory Structure

```
aPC
├── apc_train          % Unified training interface
│   └── aPC_Train.m
├── basis              % Polynomial basis construction
├── coefficients       % Regression solvers
├── dimreduct          % PLS / PCA
├── predict            % Prediction
├── UQ                 % Uncertainty quantification
├── sensitivity        % Sobol indices
├── display            % Visualization
├── utils              % Helper functions
├── benchmark          % Test functions
├── main.m             % Example script
└── README.md
```

---