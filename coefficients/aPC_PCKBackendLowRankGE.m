function results = aPC_PCKBackendLowRankGE(Psi, data_model, ModelOpts, basis_info)

    model_type = lower(strtrim(ModelOpts.ModelType));
    pck_type = lower(strtrim(ModelOpts.PCKType));

    if ~isfield(data_model, 'dY_model') || isempty(data_model.dY_model)
        error('aPC:MissingModelGradient', 'GEAPCK_LOWRANK requires prepared model-space gradients.');
    end

    if ~isfield(ModelOpts, 'KrigingOpts') || isempty(ModelOpts.KrigingOpts)
        KrigingOpts = struct();
    else
        KrigingOpts = ModelOpts.KrigingOpts;
    end

    if ~isfield(ModelOpts, 'GELowRankOpts') || isempty(ModelOpts.GELowRankOpts)
        GELowRankOpts = struct();
    else
        GELowRankOpts = ModelOpts.GELowRankOpts;
    end

    KrigingOpts = localSetDef(KrigingOpts, 'corr', 'gaussian');
    KrigingOpts = localSetDef(KrigingOpts, 'nugget', 1e-8);
    KrigingOpts = localSetDef(KrigingOpts, 'theta0', 1.0);
    KrigingOpts = localSetDef(KrigingOpts, 'theta_lb', 1e-3);
    KrigingOpts = localSetDef(KrigingOpts, 'theta_ub', 1e2);
    KrigingOpts = localSetDef(KrigingOpts, 'optimize_theta', false);
    KrigingOpts = localSetDef(KrigingOpts, 'DisplayProgress', false);
    KrigingOpts = localSetDef(KrigingOpts, 'early_stop_patience', inf);

    GELowRankOpts = localSetDef(GELowRankOpts, 'rank', 3);
    GELowRankOpts = localSetDef(GELowRankOpts, 'method', 'svd');
    GELowRankOpts.method = lower(strtrim(GELowRankOpts.method));
    if ~strcmp(GELowRankOpts.method, 'svd')
        error('aPC:UnsupportedGELowRankMethod', 'GELowRankOpts.method currently only supports ''svd''.');
    end

    trend_builder = 'lars';
    if isfield(ModelOpts, 'TrendBuilder') && ~isempty(ModelOpts.TrendBuilder)
        trend_builder = lower(strtrim(ModelOpts.TrendBuilder));
    end

    Y = data_model.Y_train(:);
    U = data_model.U;
    G = data_model.dY_model;

    basis_info = localAddUnivDerivativeCache(U, basis_info);

    trend_path_idx = localBuildTrendPath(Psi, Y, trend_builder);
    trend_path_idx = unique(trend_path_idx, 'stable');
    if isempty(trend_path_idx)
        trend_path_idx = 1:min(size(Psi, 2), 1);
    end
    final_support = trend_path_idx;
    [k_start, k_end] = localResolveTrendRange(KrigingOpts, numel(trend_path_idx));
    trend_sizes = k_start:k_end;

    best = struct();
    no_improve_count = 0;
    theta_prev = [];
    for i = 1:numel(trend_sizes)
        theta_init = [];
        if strcmp(pck_type, 'optimal') && KrigingOpts.optimize_theta && ~isempty(theta_prev)
            theta_init = theta_prev;
        end
        trend_idx = final_support(1:trend_sizes(i));
        model_i = localEvaluateTrendIdx(trend_idx, i, numel(trend_sizes), Psi, U, Y, G, basis_info, KrigingOpts, GELowRankOpts, theta_init);
        if strcmp(pck_type, 'optimal') && KrigingOpts.optimize_theta && all(isfinite(model_i.theta))
            theta_prev = model_i.theta;
        end
        if ~isfield(best, 'trend_idx') || model_i.loo < best.loo
            best = model_i;
            no_improve_count = 0;
        else
            no_improve_count = no_improve_count + 1;
            if strcmp(pck_type, 'optimal') && no_improve_count >= KrigingOpts.early_stop_patience
                break;
            end
        end
        if strcmp(pck_type, 'sequential')
            break;
        end
    end

    results = struct();
    results.ModelType = model_type;
    results.PCKType = pck_type;
    results.TrendBuilder = upper(trend_builder);
    results.PCKModel = best;
    results.coefficients = zeros(size(Psi, 2), 1);
    results.coefficients(best.trend_idx) = best.beta;
    results.TrendPathIdx = trend_path_idx;
    results.LOO = best.loo;
    results.normEmpErr = best.normEmpErr;
    results.max_score = 1 - best.loo;
    results.optErrorParams = struct('loo', best.loo, 'criterion', 'lowrank-ge-kriging-approx-loo');
    results.PCKModel.U_train = U;
    results.PCKModel.corr = KrigingOpts.corr;
    results.PCKModel.nugget = KrigingOpts.nugget;
    results.PCKModel.use_ge = true;
    results.PCKModel.mode = 'lowrank';
    results.PCKModel.lowrank_options = GELowRankOpts;
    results.basis_info = basis_info;
end

function model_i = localEvaluateTrendIdx(trend_idx, trend_pos, trend_total, Psi, U, Y, G, basis_info, KrigingOpts, GELowRankOpts, theta_init)
    F = Psi(:, trend_idx);
    dF = localBuildTrendGradientMatrix(U, basis_info, trend_idx);

    if KrigingOpts.DisplayProgress
        fprintf('%s\n', localFormatTrendProgress(trend_pos, trend_total, 30));
    end

    model_i = localFitOne(F, dF, Y, U, G, KrigingOpts, GELowRankOpts, theta_init);
    model_i.trend_idx = trend_idx;
end

function model = localFitOne(F, dF, Y, U, G, KrigingOpts, GELowRankOpts, theta_init)
    [~, d] = size(U);
    if isempty(theta_init)
        theta0 = KrigingOpts.theta0;
    else
        theta0 = theta_init;
    end
    if isscalar(theta0)
        theta0 = repmat(theta0, 1, d);
    else
        theta0 = theta0(:)';
    end
    if isscalar(KrigingOpts.theta_lb)
        theta_lb = repmat(KrigingOpts.theta_lb, 1, d);
    else
        theta_lb = KrigingOpts.theta_lb;
    end
    if isscalar(KrigingOpts.theta_ub)
        theta_ub = repmat(KrigingOpts.theta_ub, 1, d);
    else
        theta_ub = KrigingOpts.theta_ub;
    end

    if KrigingOpts.optimize_theta
        z0 = log(theta0);
        zlb = log(theta_lb);
        zub = log(theta_ub);
        opts = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', 'MaxIterations', 80, 'MaxFunctionEvaluations', 300);
        [zopt, ~, exitflag, output] = fmincon(@(z) localNLLOnly(exp(z), F, dF, Y, U, G, KrigingOpts, GELowRankOpts), ...
            z0, [], [], [], [], zlb, zub, [], opts);
        theta = exp(zopt);
        theta_optimizer = 'fmincon';
    else
        theta = theta0;
        theta_optimizer = 'fixed';
        exitflag = [];
        output = struct();
    end

    theta = min(max(theta, theta_lb), theta_ub);
    model = localBuildModel(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts);
    model.theta = theta;
    model.theta_init_used = theta0;
    model.theta_optimizer = theta_optimizer;
    model.theta_exitflag = exitflag;
    model.theta_opt_output = output;
end

function nll = localNLLOnly(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts)
    [nll, ~] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts);
end

function model = localBuildModel(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts)
    [~, core] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts);

    if core.chol_failed
        alpha = core.alpha;
        loo = inf;
        normEmpErr = inf;
    else
        alpha = core.alpha;
        residual = core.residual;
        Uinv = core.L' \ eye(size(core.L));
        Kinv_diag = sum(Uinv.^2, 2);
        Nf = size(Y, 1);
        loo_res = alpha(1:Nf) ./ max(Kinv_diag(1:Nf), eps);
        loo = mean(loo_res.^2) / max(var(Y, 1), eps);
        normEmpErr = mean(residual(1:Nf).^2) / max(var(Y, 1), eps);
    end

    model = struct();
    model.beta = core.beta;
    model.alpha = alpha;
    model.K = core.K;
    model.L = core.L;
    model.F = F;
    model.loo = loo;
    model.normEmpErr = normEmpErr;
    model.nll = core.nll;
    model.lowrank = core.lowrank;
end

function [nll, core] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, GELowRankOpts)
    [K, y_aug, H, lowrank] = pckBuildLowRankAugmentedSystem(U, Y, G, F, dF, theta, KrigingOpts.corr, KrigingOpts.nugget, GELowRankOpts.rank);
    K = (K + K') * 0.5;
    [L, p] = chol(K, 'lower');
    if p ~= 0
        nll = 1e12;
        core = localFallbackCore(F, Y, U, GELowRankOpts.rank);
        return;
    end

    KinvHY = L' \ (L \ [H, y_aug]);
    KinvH = KinvHY(:, 1:size(H, 2));
    Kinvy = KinvHY(:, end);
    A = H' * KinvH;
    b = H' * Kinvy;
    if rcond(A) < 1e-12
        beta = pinv(A) * b;
    else
        beta = A \ b;
    end

    residual = y_aug - H * beta;
    alpha = Kinvy - KinvH * beta;
    nobs = numel(y_aug);
    sigma2 = max((residual' * alpha) / nobs, eps);
    nll = nobs * log(sigma2) + 2 * sum(log(diag(L)));

    core = struct();
    core.beta = beta;
    core.alpha = alpha;
    core.K = K;
    core.L = L;
    core.F = F;
    core.residual = residual;
    core.sigma2 = sigma2;
    core.nll = nll;
    core.lowrank = lowrank;
    core.chol_failed = false;
end

function core = localFallbackCore(F, Y, U, rank)
    core = struct();
    core.beta = zeros(size(F, 2), 1);
    core.alpha = zeros(size(Y, 1) * (1 + min(rank, size(U, 2))), 1);
    core.K = eye(numel(core.alpha));
    core.L = eye(numel(core.alpha));
    core.F = F;
    core.residual = inf(size(core.alpha));
    core.sigma2 = inf;
    core.nll = inf;
    core.lowrank = struct('rank', min(rank, size(U, 2)), 'V_r', eye(size(U, 2), min(rank, size(U, 2))), 'Z', zeros(size(U, 1), min(rank, size(U, 2))), 'singular_values', zeros(min(size(U, 1), size(U, 2)), 1));
    core.chol_failed = true;
end

function S = localSetDef(S, field, val)
    if ~isfield(S, field) || isempty(S.(field))
        S.(field) = val;
    end
end

function [k_start, k_end] = localResolveTrendRange(KrigingOpts, total_terms)
    if ~isfield(KrigingOpts, 'trend_range') || isempty(KrigingOpts.trend_range)
        k_start = 1;
        k_end = total_terms;
        return;
    end

    cfg = KrigingOpts.trend_range;
    if ~isnumeric(cfg) || any(~isfinite(cfg(:)))
        error('aPC:InvalidTrendRange', 'KrigingOpts.trend_range must be numeric and finite.');
    end

    if isscalar(cfg)
        k_start = 1;
        k_end = localResolveOneBound(cfg, total_terms);
        return;
    end

    if numel(cfg) == 2
        k1 = localResolveOneBound(cfg(1), total_terms);
        k2 = localResolveOneBound(cfg(2), total_terms);
        k_start = min(k1, k2);
        k_end = max(k1, k2);
        return;
    end

    error('aPC:InvalidTrendRange', 'KrigingOpts.trend_range must be scalar or 2-element vector.');
end

function trend_path_idx = localBuildTrendPath(Psi, Y, trend_builder)
    switch lower(trend_builder)
        case 'lars'
            trend_res = aPC_LARS(Psi, Y);
            if ~isfield(trend_res, 'lars_idx') || isempty(trend_res.lars_idx)
                error('aPC:MissingLarsIdx', ...
                    'TrendBuilder ''lars'' must return a non-empty lars_idx path.');
            end
            trend_path_idx = trend_res.lars_idx(:)';
        otherwise
            error('aPC:UnsupportedTrendBuilder', ...
                'Unsupported TrendBuilder: %s. Currently only ''lars'' is supported.', trend_builder);
    end
end

function k = localResolveOneBound(v, total_terms)
    if v <= 0
        error('aPC:InvalidTrendRange', 'KrigingOpts.trend_range values must be positive.');
    end
    if v <= 1
        k = max(1, ceil(v * total_terms));
    else
        k = min(total_terms, round(v));
    end
end

function line = localFormatTrendProgress(trend_pos, trend_total, bar_len)
    pct = trend_pos / trend_total;
    filled = round(pct * bar_len);
    bar_str = [repmat('#', 1, filled), repmat('-', 1, bar_len - filled)];
    line = sprintf('[aPC_PCKBackendLowRankGE] [Trend-Serial] trend %d/%d [%s] %3d%%', ...
        trend_pos, trend_total, bar_str, round(pct * 100));
end

function dF = localBuildTrendGradientMatrix(U, basis_info, trend_idx)
    Indices = basis_info.Indices(trend_idx, :);
    univ_vals = basis_info.univ_vals;
    univ_dvals = basis_info.univ_dvals;
    N = size(U, 1);
    M = size(U, 2);
    K = size(Indices, 1);
    dF = zeros(N * M, K);
    for a = 1:M
        rows = (a - 1) * N + (1:N);
        for t = 1:K
            term_val = ones(N, 1);
            for j = 1:M
                deg = Indices(t, j);
                if j == a
                    val = univ_dvals(:, j, deg + 1);
                else
                    val = univ_vals(:, j, deg + 1);
                end
                term_val = term_val .* val;
            end
            dF(rows, t) = term_val;
        end
    end
end

function basis_info = localAddUnivDerivativeCache(U, basis_info)
    if isfield(basis_info, 'univ_dvals') && ~isempty(basis_info.univ_dvals)
        return;
    end
    poly_coeffs = basis_info.poly_coeffs;
    univ_vals = basis_info.univ_vals;
    [N, M, P1] = size(univ_vals);
    univ_dvals = zeros(N, M, P1);
    for j = 1:M
        uj = U(:, j);
        for k = 1:P1
            dcoeff = polyder(poly_coeffs{j}{k});
            if isempty(dcoeff)
                univ_dvals(:, j, k) = 0;
            else
                univ_dvals(:, j, k) = polyval(dcoeff, uj);
            end
        end
    end
    basis_info.univ_dvals = univ_dvals;
end
