# GE-aPC-K

MATLAB code for arbitrary polynomial chaos Kriging (aPCK), gradient-enhanced
aPCK, and the GRAS projected-gradient formulation.

## Requirements

MATLAB with Statistics and Machine Learning Toolbox (`zscore` for input
standardization, PLS, PCA, KPCA statistics, Ridge cross-validation, and Lasso).
Other dependencies depend on the workflow:

- Optimization Toolbox: Kriging length-scale optimization with `fmincon`.
- UQLab: input sampling with the supplied benchmark input definitions.
- Parallel Computing Toolbox: parallel searches.
- Manopt: MCP dimension reduction.

## Quick start

From the repository root in MATLAB:

```matlab
addpath('core');
aPC;
rng(1);

data.X_train = 2 * rand(30, 2) - 1;
data.Y_train = sin(data.X_train(:, 1)) + data.X_train(:, 2).^2;
data.dYdX_train = [cos(data.X_train(:, 1)), 2 * data.X_train(:, 2)];

opts.Order = 3;
opts.qnorm = 1;
opts.aPC_Method = 'stieltjes';
opts.RegressionMethod = 'lars';
opts.ModelType = 'geapck_lowrank';
opts.PCKType = 'sequential';
opts.GradientMode = 'provided';
opts.Parallel = false;
opts.KrigingOpts.optimize_theta = false;
opts.GELowRankOpts.rank = 1;
opts.GELowRankOpts.method = 'hybrid_residual_subspace';
opts.GELowRankOpts.weighted_gradient = true;
opts.GELowRankOpts.weighting_budget = 0.5;
opts.GELowRankOpts.weighting_score = 'gradnorm';

[model, opts, data, reduction] = aPC_Train(data, opts);
Xtest = 2 * rand(20, 2) - 1;
Ypred = aPC_Predict(Xtest, model, opts, reduction);
```

Inputs use one sample per row. `X_train` and `dYdX_train` are `N x D`;
`Y_train` is `N x 1`. Gradients are with respect to the original input variables.
The example uses fixed length scales and does not need UQLab or Optimization Toolbox.

`ModelType` accepts `'apc'`, `'apck'`, `'geapck'`, or `'geapck_lowrank'`.
For APCK models, `PCKType` is `'sequential'` or `'optimal'`.
`Order`, `qnorm`, and the low-rank `rank` option can be searched over candidate
values. UQ and Sobol post-processing currently support `'apc'` only.

Set `ReductionMethod` and `red_dim` for optional dimension reduction.
Supported methods are `'pca'`, `'pls'`, `'kpca'`, `'spca_hsic'`, `'skpca_hsic'`,
`'mcp'`, `'ica'`, `'rp'`, and `'as'` (active subspaces).
AS uses response gradients configured by `GradientMode`.
Gradient-enhanced models require an explicit linear map and do not support
`'kpca'` or `'skpca_hsic'`.

## Acknowledgments

The KPCA classes in `dimreduct/KPCA/` are from
[Kepeng Qiu's KPCA-MATLAB](https://github.com/iqiukp/KPCA-MATLAB)
and retain the original author notices.
The supervised KPCA module also uses its `Kernel` class.
`benchmark/wingweight.m` is by Sonja Surjanovic and Derek Bingham and carries
its original GPL v2 notice.
