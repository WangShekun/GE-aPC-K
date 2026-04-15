function results = aPC_PCKBackendBlock(Psi, data_model, ModelOpts, basis_info)

    model_type = lower(strtrim(ModelOpts.ModelType));
    pck_type = lower(strtrim(ModelOpts.PCKType));

    if ~isfield(data_model, 'dY_model') || isempty(data_model.dY_model)
        error('aPC:MissingModelGradient', 'GEAPCK_BLOCK requires prepared model-space gradients.');
    end

    if ~isfield(ModelOpts, 'KrigingOpts') || isempty(ModelOpts.KrigingOpts)
        KrigingOpts = struct();
    else
        KrigingOpts = ModelOpts.KrigingOpts;
    end

    if ~isfield(ModelOpts, 'GEBlockOpts') || isempty(ModelOpts.GEBlockOpts)
        GEBlockOpts = struct();
    else
        GEBlockOpts = ModelOpts.GEBlockOpts;
    end

    KrigingOpts = localSetDef(KrigingOpts, 'corr', 'gaussian');
    KrigingOpts = localSetDef(KrigingOpts, 'nugget', 1e-8);
    KrigingOpts = localSetDef(KrigingOpts, 'theta0', 1.0);
    KrigingOpts = localSetDef(KrigingOpts, 'theta_lb', 1e-3);
    KrigingOpts = localSetDef(KrigingOpts, 'theta_ub', 1e2);
    KrigingOpts = localSetDef(KrigingOpts, 'optimize_theta', false);
    KrigingOpts = localSetDef(KrigingOpts, 'DisplayProgress', false);
    KrigingOpts = localSetDef(KrigingOpts, 'early_stop_patience', inf);

    GEBlockOpts = localSetDef(GEBlockOpts, 'num_blocks', 4);
    if ~isscalar(GEBlockOpts.num_blocks) || ~isfinite(GEBlockOpts.num_blocks) || GEBlockOpts.num_blocks < 1
        error('aPC:InvalidGEBlockCount', 'GEBlockOpts.num_blocks must be a finite scalar >= 1.');
    end

    parallel_scope = 'serial';
    if isfield(ModelOpts, 'ParallelScope') && ~isempty(ModelOpts.ParallelScope)
        parallel_scope = lower(strtrim(ModelOpts.ParallelScope));
    end

    trend_builder = 'lars';
    if isfield(ModelOpts, 'TrendBuilder') && ~isempty(ModelOpts.TrendBuilder)
        trend_builder = lower(strtrim(ModelOpts.TrendBuilder));
    end

    Y = data_model.Y_train(:);
    U = data_model.U;
    G = data_model.dY_model;
    [N, d] = size(U);

    basis_info = localAddUnivDerivativeCache(U, basis_info);
    block_model = localBuildBlocks(U, G, min(max(1, round(GEBlockOpts.num_blocks)), N));

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
        model_i = localEvaluateTrendIdx(trend_idx, i, numel(trend_sizes), Psi, U, Y, G, basis_info, KrigingOpts, block_model, theta_init);
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
    results.optErrorParams = struct('loo', best.loo, 'criterion', 'block-train-error');
    results.PCKModel.U_train = U;
    results.PCKModel.corr = KrigingOpts.corr;
    results.PCKModel.nugget = KrigingOpts.nugget;
    results.PCKModel.use_ge = true;
    results.PCKModel.parallel_scope = parallel_scope;
    results.PCKModel.mode = 'block';
    results.PCKModel.block_options = GEBlockOpts;
    results.basis_info = basis_info;
end

function model_i = localEvaluateTrendIdx(trend_idx, trend_pos, trend_total, Psi, U, Y, G, basis_info, KrigingOpts, block_model, theta_init)
    F = Psi(:, trend_idx);
    dF = localBuildTrendGradientMatrix(U, basis_info, trend_idx);

    if KrigingOpts.DisplayProgress
        fprintf('%s\n', localFormatTrendProgress(trend_pos, trend_total, 30));
    end

    model_i = localFitOne(F, dF, Y, U, G, KrigingOpts, block_model, theta_init);
    model_i.trend_idx = trend_idx;
end

function model = localFitOne(F, dF, Y, U, G, KrigingOpts, block_model, theta_init)
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
        [zopt, ~, exitflag, output] = fmincon(@(z) localNLLOnly(exp(z), F, dF, Y, U, G, KrigingOpts, block_model), ...
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
    model = localBuildModel(theta, F, dF, Y, U, G, KrigingOpts, block_model);
    model.theta = theta;
    model.theta_init_used = theta0;
    model.theta_optimizer = theta_optimizer;
    model.theta_exitflag = exitflag;
    model.theta_opt_output = output;
end

function nll = localNLLOnly(theta, F, dF, Y, U, G, KrigingOpts, block_model)
    [nll, ~] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, block_model);
end

function model = localBuildModel(theta, F, dF, Y, U, G, KrigingOpts, block_model)
    [~, core] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, block_model);
    if core.chol_failed
        loo = inf;
        normEmpErr = inf;
    else
        Uinv = core.Lff' \ eye(size(core.Lff));
        Kinv_diag = sum(Uinv .^ 2, 2);
        loo_res = core.alpha ./ max(Kinv_diag, eps);
        loo = mean(loo_res .^ 2) / max(var(Y, 1), eps);
        normEmpErr = mean(core.residual .^ 2) / max(var(Y, 1), eps);
    end

    model = struct();
    model.beta = core.beta;
    model.alpha = core.alpha;
    model.K = core.Kff;
    model.L = core.Lff;
    model.F = F;
    model.loo = loo;
    model.normEmpErr = normEmpErr;
    model.blocks = core.blocks;
    model.tau_dist = core.tau_dist;
end

function [nll, core] = localEvalThetaCore(theta, F, dF, Y, U, G, KrigingOpts, block_model)
    [Kff_raw, Kfg_blocks, Kgg_blocks] = pckEvalKernel(U, U, theta, KrigingOpts.corr);
    if isempty(Kfg_blocks)
        error('aPC:UnsupportedGECorrelation', 'GEAPCK_BLOCK requires differentiable kernel: %s', KrigingOpts.corr);
    end

    Kff = Kff_raw + KrigingOpts.nugget * eye(size(Kff_raw));
    Kff = (Kff + Kff') * 0.5;
    [Lff, p] = chol(Kff, 'lower');
    if p ~= 0
        nll = 1e12;
        core = localFallbackCore(F, Y, U, block_model);
        return;
    end

    KinvFY = Lff' \ (Lff \ [F, Y]);
    KinvF = KinvFY(:, 1:size(F, 2));
    Kinvy = KinvFY(:, end);
    A = F' * KinvF;
    b = F' * Kinvy;
    if rcond(A) < 1e-12
        beta = pinv(A) * b;
    else
        beta = A \ b;
    end

    residual_y = Y - F * beta;
    alpha = Kinvy - KinvF * beta;
    sigma2_f = max((residual_y' * alpha) / numel(Y), eps);
    nll = numel(Y) * log(sigma2_f) + 2 * sum(log(diag(Lff)));

    blocks = block_model.blocks;
    for bidx = 1:numel(blocks)
        block = blocks{bidx};
        [Kfg_b, Kgf_b, Kgg_b, g_b, dF_b] = localExtractBlock(block, Kfg_blocks, Kgg_blocks, G, dF, KrigingOpts.nugget);
        C_b = Lff' \ (Lff \ Kfg_b);
        S_b = Kgg_b - Kgf_b * C_b;
        S_b = (S_b + S_b') * 0.5;
        [Lb, pb] = chol(S_b, 'lower');
        if pb ~= 0
            nll = 1e12;
            core = localFallbackCore(F, Y, U, block_model);
            return;
        end
        residual_b = g_b - dF_b * beta - Kgf_b * alpha;
        eta_b = Lb' \ (Lb \ residual_b);
        sigma2_b = max((residual_b' * eta_b) / numel(g_b), eps);
        nll = nll + numel(g_b) * log(sigma2_b) + 2 * sum(log(diag(Lb)));
        blocks{bidx}.eta = eta_b;
        blocks{bidx}.C = C_b;
    end

    core = struct();
    core.beta = beta;
    core.alpha = alpha;
    core.Kff = Kff;
    core.Lff = Lff;
    core.F = F;
    core.residual = residual_y;
    core.blocks = blocks;
    core.tau_dist = block_model.tau_dist;
    core.chol_failed = false;
end

function [Kfg_b, Kgf_b, Kgg_b, g_b, dF_b] = localExtractBlock(block, Kfg_blocks, Kgg_blocks, G, dF, nugget)
    N = size(G, 1);
    d = size(G, 2);
    sample_idx = block.sample_idx;
    Kfg_cells = cell(1, d);
    for a = 1:d
        Kfg_cells{a} = Kfg_blocks{a}(:, sample_idx);
    end
    Kfg_b = cat(2, Kfg_cells{:});
    Kgf_b = -Kfg_b';

    m = numel(sample_idx);
    Kgg_b = zeros(m * d, m * d);
    for a = 1:d
        rows = (a - 1) * m + (1:m);
        for b = 1:d
            cols = (b - 1) * m + (1:m);
            Kgg_b(rows, cols) = Kgg_blocks{a, b}(sample_idx, sample_idx);
        end
    end
    Kgg_b(1:size(Kgg_b, 1) + 1:end) = Kgg_b(1:size(Kgg_b, 1) + 1:end) + nugget;

    g_b = reshape(G(sample_idx, :), [], 1);
    dF_b = dF(block.grad_rows, :);
end

function core = localFallbackCore(F, Y, U, block_model)
    core = struct();
    core.beta = zeros(size(F, 2), 1);
    core.alpha = zeros(size(Y, 1), 1);
    core.Kff = eye(size(U, 1));
    core.Lff = eye(size(U, 1));
    core.F = F;
    core.residual = inf(size(Y));
    core.blocks = localEmptyBlockParams(block_model.blocks, size(U, 1));
    core.tau_dist = block_model.tau_dist;
    core.chol_failed = true;
end

function blocks = localEmptyBlockParams(blocks, N)
    for i = 1:numel(blocks)
        block_dim = numel(blocks{i}.sample_idx) * size(blocks{i}.center, 2);
        blocks{i}.eta = zeros(block_dim, 1);
        blocks{i}.C = zeros(N, block_dim);
    end
end

function block_model = localBuildBlocks(U, G, num_blocks)
    N = size(U, 1);
    g_norm = sqrt(sum(G .^ 2, 2));
    g_dir = G ./ max(g_norm, eps);
    features = [U, log(g_norm + eps), g_dir];
    features = localStandardize(features);
    assign = localKMeansAssign(features, num_blocks);
    blocks = cell(1, num_blocks);
    for bidx = 1:num_blocks
        sample_idx = find(assign == bidx);
        if isempty(sample_idx)
            sample_idx = bidx;
        end
        grad_rows = localBlockRows(sample_idx(:)', size(U, 1), size(U, 2));
        blocks{bidx} = struct('sample_idx', sample_idx(:)', 'grad_rows', grad_rows, 'center', mean(U(sample_idx, :), 1));
    end
    tau_dist = localResolveTau(blocks);
    block_model = struct('blocks', {blocks}, 'tau_dist', tau_dist);
end

function Xs = localStandardize(X)
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = 1;
    Xs = (X - mu) ./ sigma;
end

function assign = localKMeansAssign(X, K)
    opts = statset('MaxIter', 200, 'Display', 'off');
    assign = kmeans(X, K, 'Distance', 'sqeuclidean', 'Replicates', 5, 'Options', opts, 'EmptyAction', 'singleton');
end

function D = localPairwiseSqDist(A, B)
    aa = sum(A .^ 2, 2);
    bb = sum(B .^ 2, 2)';
    D = max(aa + bb - 2 * (A * B'), 0);
end

function rows = localBlockRows(sample_idx, N, d)
    rows = zeros(numel(sample_idx) * d, 1);
    pos = 1;
    for a = 1:d
        span = pos:(pos + numel(sample_idx) - 1);
        rows(span) = (a - 1) * N + sample_idx(:);
        pos = pos + numel(sample_idx);
    end
end

function tau_dist = localResolveTau(blocks)
    if numel(blocks) <= 1
        tau_dist = 1;
        return;
    end
    centers = cell2mat(cellfun(@(b) b.center, blocks, 'UniformOutput', false)');
    D = localPairwiseSqDist(centers, centers);
    D = D(triu(true(size(D)), 1));
    D = D(D > 0);
    if isempty(D)
        tau_dist = 1;
    else
        tau_dist = sqrt(median(D));
    end
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
                error('aPC:MissingLarsIdx', 'TrendBuilder ''lars'' must return a non-empty lars_idx path.');
            end
            trend_path_idx = trend_res.lars_idx(:)';
        otherwise
            error('aPC:UnsupportedTrendBuilder', 'Unsupported TrendBuilder: %s. Currently only ''lars'' is supported.', trend_builder);
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
    line = sprintf('[aPC_PCKBackendBlock] [Trend-Serial] trend %d/%d [%s] %3d%%', ...
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
