function [X_red, ReduceModel] = aPC_dimReduction_ge_pls(U, Y, dYdU, ncomp, ModelOpts)
% Gradient-enhanced PLS reduction using an augmented response matrix.
% Method options are read from ModelOpts.ge_pls_opts (recommended).

    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', 'ge_pls requires gradients in U-space.');
    end

    opts = struct();
    if nargin >= 5 && isfield(ModelOpts, 'ge_pls_opts') && ~isempty(ModelOpts.ge_pls_opts)
        opts = ModelOpts.ge_pls_opts;
    end
    opts = localSetDefault(opts, 'grad_weight', 1.0);

    grad_weight = opts.grad_weight;

    Y_aug = [local_zscore_cols(Y), grad_weight * local_zscore_cols(dYdU)];
    [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = ...
        plsregress(U, Y_aug, ncomp, 'Intercept', false);

    params = struct();
    params.W = stats.W;
    params.grad_weight = grad_weight;

    ReduceModel = struct();
    ReduceModel.method = 'ge_pls';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) U_new * params.W;

    info = struct();
    info.XL = XL;
    info.YL = YL;
    info.XS = XS;
    info.YS = YS;
    info.BETA = BETA;
    info.PCTVAR = PCTVAR;
    info.MSE = MSE;
    info.stats = stats;
    ReduceModel.info = info;

    X_red = XS;
end

function Z = local_zscore_cols(X)
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = 1;
    Z = (X - mu) ./ sigma;
end

function opts = localSetDefault(opts, fieldName, defaultVal)
    if ~isfield(opts, fieldName) || isempty(opts.(fieldName))
        opts.(fieldName) = defaultVal;
    end
end
