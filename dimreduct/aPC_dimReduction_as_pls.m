function [X_red, ReduceModel] = aPC_dimReduction_as_pls(U, Y, dYdU, ncomp, ModelOpts)
% Active-subspace-plus-PLS reduction using gradient covariance first and PLS second.
% Method options are read from ModelOpts.as_pls_opts (recommended).

    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', 'as_pls requires gradients in U-space.');
    end

    opts = struct();
    if nargin >= 5 && isfield(ModelOpts, 'as_pls_opts') && ~isempty(ModelOpts.as_pls_opts)
        opts = ModelOpts.as_pls_opts;
    end

    [~, M] = size(U);
    as_dim = min(M, max(ncomp, 2 * ncomp));
    if isfield(opts, 'as_dim') && ~isempty(opts.as_dim)
        as_dim = opts.as_dim;
    end
    if as_dim < ncomp
        error('aPC:InvalidASDim', 'as_dim must be at least red_dim for as_pls.');
    end

    C = (dYdU' * dYdU) / max(size(dYdU, 1), 1);
    C = (C + C') * 0.5;
    [V, D] = eig(C);
    [eigvals, idx] = sort(diag(D), 'descend');
    A_as = V(:, idx(1:as_dim));

    U_as = U * A_as;
    [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = ...
        plsregress(U_as, Y, ncomp, 'Intercept', false);

    params = struct();
    params.A_as = A_as;
    params.W = stats.W;
    params.A = A_as * stats.W;
    params.eigvals = eigvals;
    params.as_dim = as_dim;

    ReduceModel = struct();
    ReduceModel.method = 'as_pls';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) U_new * params.A;

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
