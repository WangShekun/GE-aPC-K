
---

# **CHANGELOG**

## **v1.1.0**

1. **Added automatic model-structure selection based on LOO error**, covering
   polynomial order, q-norm truncation, and reduced dimensionality.

2. **Added an option to switch between LARS entering-order basis selection and the traditional nonzero-index basis set**, with consistent handling of the constant term.

3. **Added nonlinear dimensionality reduction via KPCA**,
   expanding the available reduction methods beyond PLS and PCA.

4. **Unified the dimensionality-reduction workflow**,
   introducing a consistent two-stage process across all methods:

   * a second standardization is applied after reduction,
   * and this transformation is embedded into `ReduceModel.transform` to ensure that prediction follows exactly the same mapping as training.

5. **Improved training progress display**,
   including a dynamic, single-line progress bar with real-time parameter updates.

6. **Improved code modularity for future hyperparameter auto-search**,
   allowing new tunable fields (e.g., kernel parameters or additional reduction settings) to be integrated cleanly without modifying the core training logic.

7. **Added support for parallel hyperparameter search**,
   enabling multi-core evaluation of all candidate configurations to significantly reduce total training time.


### **KPCA Input Specification**

When `ModelOpts.ReductionMethod = 'kpca'` is used, KPCA requires only two inputs:

1. **`ModelOpts.red_dim`** — dimensionality to retain

2. **`ModelOpts.kernel`** — kernel object defining the nonlinear mapping.
   The kernel must be constructed using the `Kernel` class, and the following types and parameters are supported:

   * **Linear:**
     `Kernel('type','linear')`
   * **Polynomial** (parameters: `degree`, `gamma`, `offset`):
     `Kernel('type','polynomial','degree',d,'gamma',g,'offset',c)`
   * **Gaussian** (parameter: `gamma`):
     `Kernel('type','gaussian','gamma',g)`
   * **Sigmoid** (parameters: `gamma`, `offset`):
     `Kernel('type','sigmoid','gamma',g,'offset',c)`
   * **Laplacian** (parameter: `gamma`):
     `Kernel('type','laplacian','gamma',g)`

This completes all required user inputs for KPCA.

---

## **v1.0.0 — Initial Release**

This version provides the complete foundation of the **arbitrary Polynomial Chaos Expansion (aPCE)** framework, including polynomial construction, preprocessing, dimensionality reduction, regression modeling, prediction, uncertainty quantification, and sensitivity analysis.

### **aPCE Polynomial System**

* Supports two univariate polynomial generation approaches (2012 and 2018 versions)
* Moment-based orthogonal / orthonormal polynomial construction
* Multi-index generation with q-norm truncation
* Tensor-product multivariate polynomial basis

### **Data Preprocessing**

* Standardization using sample mean and standard deviation
* Automatic removal of zero-variance dimensions
* Consistent transformations for both training and prediction

### **Dimensionality Reduction**

* Supports PLS reduction
* Supports PCA reduction
* Reduced representation fully integrated with PCE basis construction and regression
* Shared projection used during both training and prediction

### **Regression Modeling**

Provides multiple regression methods for PCE coefficient estimation:

* OLS
* LARS
* Ridge
* Lasso

Suitable for both sparse and dense aPCE models.

### **Prediction**

* Applies the same standardization parameters used during training
* Automatically applies dimensionality reduction
* Reconstructs polynomial bases and produces model predictions

### **Uncertainty Quantification (UQ)**

Includes statistical outputs derived from the aPCE model:

* Mean
* Variance
* Standard deviation
* Skewness
* Kurtosis
* Confidence interval (CI)
* VaR / CVaR

### **Sobol Sensitivity Analysis**

Supports:

* First-order indices
* Second-order indices
* Total-effect indices
  With visualization support.

### **Visualization**

Includes:

* UQ distribution plots
* Sobol index plots
* PCE diagnostic plots

---