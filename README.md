# Arbitrary Polynomial Chaos Expansion (aPCE) Framework

This repository is a modular MATLAB implementation of arbitrary Polynomial Chaos Expansion for:

- surrogate modeling,
- uncertainty quantification (UQ),
- global sensitivity analysis (GSA),
- and hybrid PCE-Kriging variants (APCK / GEAPCK).

It supports automatic hyperparameter search, multiple regression backends, optional dimension reduction, and unified prediction.

## Quick Start

```matlab
repo_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, 'core'));
aPC;

run("main.m")           % single-run script
% or
run("main_compare.m")   % benchmark-style model comparison
```

## End-to-End Pipeline

Training (`aPC_Train`):

1. Standardize `X_train` to `U` (`aPC_PCE_map_to_U`)
2. Optional dimension reduction (`aPC_dimReduction`)
3. Build multi-index terms (`aPC_GenerateTerms`)
4. Build polynomial basis matrix `Psi`
5. Fit model (APC or APCK/GEAPCK backend)
6. Select best config by LOO over search grid

Prediction (`aPC_Predict`):

1. Apply saved standardization mapping
2. Apply saved reduction transform (if any)
3. Rebuild basis
4. Predict with APC or PCK backend

## Model Families

- `apc`: pure arbitrary PCE regression
- `apck`: PCE trend + Kriging residual correction
- `geapck`: gradient-enhanced APCK

Important:

- `PCKType` is required for `apck` and `geapck` (`sequential` or `optimal`)
- `aPC_ComputeUQ` and `aPC_ComputeSensitivity` currently support `apc` only

## Full Configuration Reference (`ModelOpts`)

All options below are supported by current code paths.

### 1) Core PCE Options

- `Order` (numeric scalar or vector)
  - maximum total polynomial degree
- `qnorm` (numeric scalar or vector)
  - q-norm truncation parameter for multi-index generation
- `aPC_Method` (string)
  - supported: `'2012'`, `'2018'`, `'stieltjes'`
- `RegressionMethod` (string, case-insensitive)
  - supported: `'ols'`, `'lars'`, `'lasso'`, `'ridge'`, `'omp'`
- `Parallel` (0/1)
  - whether to evaluate search candidates in parallel

### 2) Model-Type Options

- `ModelType` (required)
  - `'apc'` | `'apck'` | `'geapck'`
- `PCKType` (required when `ModelType` is `apck` or `geapck`)
  - `'sequential'` | `'optimal'`
- `GradientMode` (GEAPCK only)
  - `'provided'` | `'finite-difference'` | `'auto'`

### 3) Dimension Reduction Options

- `ReductionMethod`
  - supported: `'pls'`, `'pca'`, `'kpca'`, `'spca_hsic'`, `'skpca_hsic'`, `'mcp'`, `'ica'`, `'rp'`
- `red_dim`
  - target reduced dimension
  - `kpca` / `skpca_hsic`: supports integer `>1` or ratio `(0,1)` for explained-components mode

Note:

- After reduction, framework performs second standardization automatically and reuses it in prediction.

### 4) Kernel Options (KPCA / SKPCA-HSIC)

- Input kernel (X-space):
  - `kernel_type`
  - `kernel_gamma`
  - `kernel_degree`
  - `kernel_offset`
- Label kernel (Y-space, supervised KPCA):
  - `label_kernel_type`
  - `label_kernel_gamma`
  - `label_kernel_degree`
  - `label_kernel_offset`

### 5) MCP Options (`mcp_opts`)

- `mcp_opts.maxIter`
- `mcp_opts.whiten`
- `mcp_opts.alpha_cov`
- `mcp_opts.max_moment_order`
- `mcp_opts.moment_base`
- `mcp_opts.moment_decay`
- `mcp_opts.lambda_y`
- `mcp_opts.supervision_mode`
- `mcp_opts.manopt_solver`

### 6) ICA Options

- `ica_approach`: `'symm'` | `'defl'`
- `ica_fun`: `'tanh'` | `'gauss'` | `'pow3'`
- `ica_a1`, `ica_a2`
- `ica_epsilon`
- `ica_maxIter`
- `ica_mu`
- `ica_stabilization`: `'on'` | `'off'`
- `ica_whiten_solver`: `'svd'` | `'eig'`
- `ica_seed`
- `ica_verbose`

### 7) Random Projection (RP) Options

- `rp_type`: `'gaussian'` | `'rademacher'`
- `rp_scale`: `'jl'` | `'none'`
- `rp_seed`

### 8) Kriging Options (`KrigingOpts`) for APCK/GEAPCK

- `KrigingOpts.corr`: `'gaussian'` | `'exponential'` | `'matern32'`
  - GEAPCK currently restricts correlation to `'gaussian'`
- `KrigingOpts.nugget`
- `KrigingOpts.theta0`
- `KrigingOpts.theta_lb`
- `KrigingOpts.theta_ub`
- `KrigingOpts.optimize_theta` (true/false)
- `KrigingOpts.max_trend_terms`
  - scalar or two-element vector
  - if value `<=1`, interpreted as fraction of LARS trend path
  - if value `>1`, interpreted as absolute term count
- `KrigingOpts.DisplayProgress` (true/false)

### 9) Finite-Difference Gradient Options (`FDOpts`)

- `FDOpts.type`: `'forward'` | `'central'`
- `FDOpts.step`: relative perturbation size

Used when `GradientMode` is `'finite-difference'` or `'auto'` without provided gradients.

## Hyperparameter Auto-Search

`aPC_Train` accepts scalar-or-vector values for searchable fields and builds a Cartesian grid.

Search-enabled fields include:

- `Order`
- `qnorm`
- `red_dim`
- `kernel_type`, `kernel_gamma`, `kernel_degree`, `kernel_offset`
- `label_kernel_type`, `label_kernel_gamma`, `label_kernel_degree`, `label_kernel_offset`
- `KrigingOpts.corr`, `KrigingOpts.nugget`

Best model is selected by minimal LOO criterion.

## Data Requirements

Minimal training data structure:

- `data.X_train` (`N x M`)
- `data.Y_train` (`N x 1`)

Optional but commonly used:

- `data.X_val`, `data.Y_val` for external evaluation
- `data.dYdX_train` for GEAPCK when using provided gradients
- `data.ModelFcn` function handle for finite-difference gradient fallback

## Gradient Modes and Practical Use

For GEAPCK, three gradient sources are supported:

- `provided`: use `data.dYdX_train`
- `finite-difference`: compute gradients by repeated model evaluations
- `auto`: use provided if present, otherwise finite difference

This is exposed in `main_compare.m` via benchmark controls.

## Dimension Reduction + GEAPCK Gradient Propagation

GEAPCK requires chain-rule propagation through standardization and reduction transforms.

Currently supported for GE gradient propagation with explicit linear map:

- `pls`, `pca`, `rp`, `mcp`, `ica`

Nonlinear kernel reductions (`kpca`, `skpca_hsic`) are not currently supported for GE gradient propagation.

## Benchmark Utilities

`main_compare.m` supports side-by-side comparison of:

- UQLab PCE
- APC
- APCK
- GEAPCK

Benchmark gradient helpers are under:

- `benchmark/gradients/ishigami_grad.m` (analytic)
- `benchmark/gradients/wingweight_grad.m` (analytic)
- `benchmark/gradients/morris10_grad.m` (analytic)

## UQ and Sensitivity

- UQ: `aPC_ComputeUQ`
  - mean, variance, std, CI95, skewness, kurtosis, VaR95, CVaR95
- Sensitivity: `aPC_ComputeSensitivity`
  - first-order, second-order, total Sobol indices

At present these two modules are restricted to `ModelType='apc'`.

## Directory Structure

```text
aPC
├── apc_train
├── auto_search
├── basis
├── benchmark
│   ├── gradients
│   └── inputopts
├── coefficients
├── core
├── dimreduct
├── display
├── predict
├── sensitivity
├── tests
├── UQ
├── utils
├── main.m
├── main_compare.m
└── README.md
```

## Typical Usage Snippet

```matlab
ModelOpts = struct();
ModelOpts.Order = [3 4 5];
ModelOpts.qnorm = [0.7 1.0];
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'auto';
ModelOpts.FDOpts.type = 'central';
ModelOpts.FDOpts.step = 1e-6;

ModelOpts.ReductionMethod = 'pls';
ModelOpts.red_dim = [4 6 8];

ModelOpts.KrigingOpts.corr = 'gaussian';
ModelOpts.KrigingOpts.nugget = [1e-8 1e-6];
ModelOpts.KrigingOpts.theta0 = 1.0;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 1e2;
ModelOpts.KrigingOpts.max_trend_terms = [0.1, 0.8];
ModelOpts.KrigingOpts.optimize_theta = true;

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
y_pred = aPC_Predict(X_new, results, ModelOptsOut, ReduceModel);
```
