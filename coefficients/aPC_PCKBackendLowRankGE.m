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
    KrigingOpts = localSetDef(KrigingOpts, 'stabilization', 'nugget');
    KrigingOpts.stabilization = lower(strtrim(KrigingOpts.stabilization));
    if ~any(strcmp(KrigingOpts.stabilization, {'nugget', 'constrained_projection'}))
        error('aPC:UnsupportedStabilization', ...
            'KrigingOpts.stabilization must be ''nugget'' or ''constrained_projection''.');
    end

    GELowRankOpts = localSetDef(GELowRankOpts, 'rank', 3);
    GELowRankOpts = localSetDef(GELowRankOpts, 'method', 'svd');
    GELowRankOpts = localSetDef(GELowRankOpts, 'residual_rank', []);
    GELowRankOpts = localSetDef(GELowRankOpts, 'hybrid_residual_mode', 'post_project');
    GELowRankOpts = localSetDef(GELowRankOpts, 'operator_eta_grid', [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]);
    GELowRankOpts = localSetDef(GELowRankOpts, 'operator_grad_weight', 0.1);
    GELowRankOpts = localSetDef(GELowRankOpts, 'operator_search_method', 'grid');
    GELowRankOpts = localSetDef(GELowRankOpts, 'weighted_gradient', false);
    GELowRankOpts = localSetDef(GELowRankOpts, 'weighting_score', 'gradnorm');
    GELowRankOpts = localSetDef(GELowRankOpts, 'weighting_budget', 1.0);
    GELowRankOpts.method = lower(strtrim(GELowRankOpts.method));
    GELowRankOpts.hybrid_residual_mode = lower(strtrim(GELowRankOpts.hybrid_residual_mode));
    GELowRankOpts.operator_search_method = lower(strtrim(GELowRankOpts.operator_search_method));
    GELowRankOpts.weighting_score = lower(strtrim(GELowRankOpts.weighting_score));
    if ~any(strcmp(GELowRankOpts.method, {'svd', 'hybrid_residual_subspace', 'supervised_joint_operator'}))
        error('aPC:UnsupportedGELowRankMethod', ...
            'GELowRankOpts.method currently only supports ''svd'', ''hybrid_residual_subspace'', or ''supervised_joint_operator''.');
    end
    if ~any(strcmp(GELowRankOpts.operator_search_method, {'grid', 'fminbnd'}))
        error('aPC:UnsupportedLowRankOperatorSearchMethod', ...
            'GELowRankOpts.operator_search_method must be ''grid'' or ''fminbnd''.');
    end
    if ~any(strcmp(GELowRankOpts.hybrid_residual_mode, {'post_project', 'project_then_svd'}))
        error('aPC:UnsupportedHybridResidualMode', ...
            'GELowRankOpts.hybrid_residual_mode must be ''post_project'' or ''project_then_svd''.');
    end
    if ~any(strcmp(GELowRankOpts.weighting_score, {'gradnorm', 'residual_gradnorm'}))
        error('aPC:UnsupportedWeightedGradientScore', ...
            'GELowRankOpts.weighting_score must be ''gradnorm'' or ''residual_gradnorm''.');
    end
    trend_builder = 'lars';
    if isfield(ModelOpts, 'TrendBuilder') && ~isempty(ModelOpts.TrendBuilder)
        trend_builder = lower(strtrim(ModelOpts.TrendBuilder));
    end

    Y = data_model.Y_train(:);
    U = data_model.U;
    G = data_model.dY_model;

    trend_path_idx = localBuildTrendPath(Psi, Y, trend_builder);
    trend_path_idx = unique(trend_path_idx, 'stable');
    if isempty(trend_path_idx)
        trend_path_idx = 1:min(size(Psi, 2), 1);
    end
    basis_info = localAddUnivDerivativeCache(U, basis_info);
    lowrank_cache = localBuildLowRankCache(G, Y, Psi, trend_path_idx, U, basis_info, size(U, 2), GELowRankOpts);

    final_support = trend_path_idx;
    [k_start, k_end] = localResolveTrendRange(KrigingOpts, numel(trend_path_idx));
    trend_sizes = k_start:k_end;

    best = struct();
    if strcmp(pck_type, 'sequential')
        best = localEvaluateTrendIdx(final_support, 1, 1, Psi, U, Y, basis_info, KrigingOpts, GELowRankOpts, [], lowrank_cache);
    else
        no_improve_count = 0;
        theta_prev = [];
        for i = 1:numel(trend_sizes)
            theta_init = [];
            if strcmp(pck_type, 'optimal') && KrigingOpts.optimize_theta && ~isempty(theta_prev)
                theta_init = theta_prev;
            end
            trend_idx = final_support(1:trend_sizes(i));
            model_i = localEvaluateTrendIdx(trend_idx, i, numel(trend_sizes), Psi, U, Y, basis_info, KrigingOpts, GELowRankOpts, theta_init, lowrank_cache);
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

function model_i = localEvaluateTrendIdx(trend_idx, trend_pos, trend_total, Psi, U, Y, basis_info, KrigingOpts, GELowRankOpts, theta_init, lowrank_cache)
    F = Psi(:, trend_idx);
    dF = localBuildTrendGradientMatrix(U, basis_info, trend_idx);
    dF_r = localProjectGradientTrend(dF, lowrank_cache.V_r);

    if KrigingOpts.DisplayProgress
        fprintf('%s\n', localFormatTrendProgress(trend_pos, trend_total, 30));
    end

    model_i = localFitOne(F, dF_r, Y, U, KrigingOpts, GELowRankOpts, theta_init, lowrank_cache);
    model_i.trend_idx = trend_idx;
end

function model = localFitOne(F, dF_r, Y, U, KrigingOpts, GELowRankOpts, theta_init, lowrank_cache)
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

    if strcmp(KrigingOpts.stabilization, 'constrained_projection')
        H = [F; dF_r];
        W = null(H')';
    else
        W = [];
    end

    if KrigingOpts.optimize_theta
        z0 = log(theta0);
        zlb = log(theta_lb);
        zub = log(theta_ub);
        opts = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', 'MaxIterations', 80, 'MaxFunctionEvaluations', 300);
        [zopt, ~, exitflag, output] = fmincon(@(z) localNLLOnly(exp(z), F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W), ...
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
    model = localBuildModel(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W);
    model.theta = theta;
    model.theta_init_used = theta0;
    model.theta_optimizer = theta_optimizer;
    model.theta_exitflag = exitflag;
    model.theta_opt_output = output;
end

function nll = localNLLOnly(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W)
    [nll, ~] = localEvalThetaCore(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W);
end

function model = localBuildModel(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W)
    [~, core] = localEvalThetaCore(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W);

    if core.chol_failed
        alpha = core.alpha;
        loo = inf;
        normEmpErr = inf;
    else
        alpha = core.alpha;
        residual = core.residual;
        if strcmp(KrigingOpts.stabilization, 'constrained_projection')
            projected_inverse = core.L \ core.W;
            Kinv_diag = sum(projected_inverse.^2, 1)';
        else
            Uinv = core.L' \ eye(size(core.L));
            Kinv_diag = sum(Uinv.^2, 2);
        end
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

function [nll, core] = localEvalThetaCore(theta, F, dF_r, Y, U, KrigingOpts, GELowRankOpts, lowrank_cache, W)
    use_projection = strcmp(KrigingOpts.stabilization, 'constrained_projection');
    applied_nugget = KrigingOpts.nugget;
    if use_projection
        applied_nugget = 0;
    end
    [K, y_aug, H, lowrank] = pckBuildLowRankAugmentedSystem(U, Y, [], F, [], theta, KrigingOpts.corr, applied_nugget, GELowRankOpts.rank, lowrank_cache, dF_r);
    K = (K + K') * 0.5;
    if use_projection
        Ksolve = W * K * W';
        Ksolve = (Ksolve + Ksolve') * 0.5;
        rhs = W * y_aug;
    else
        Ksolve = K;
        rhs = [H, y_aug];
    end
    [L, p] = chol(Ksolve, 'lower');
    if p ~= 0
        nll = 1e12;
        core = localFallbackCore(F, Y, U, GELowRankOpts.rank);
        return;
    end

    if use_projection
        gamma = L' \ (L \ rhs);
        alpha = W' * gamma;
        beta = lsqminnorm(H, y_aug - K * alpha);
        residual = y_aug - H * beta;
        nobs = numel(rhs);
        sigma2 = max((rhs' * gamma) / nobs, eps);
    else
        KinvHY = L' \ (L \ rhs);
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
    end
    nll = nobs * log(sigma2) + 2 * sum(log(diag(L)));

    core = struct();
    core.beta = beta;
    core.alpha = alpha;
    core.K = K;
    core.L = L;
    core.W = W;
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

function lowrank = localBuildLowRankCache(G, Y, Psi, trend_path_idx, U, basis_info, d, GELowRankOpts)
    rank = GELowRankOpts.rank;
    if ~isscalar(rank) || ~isfinite(rank) || rank < 1
        error('aPC:InvalidGELowRankRank', 'GELowRankOpts.rank must be a finite scalar >= 1.');
    end
    r = min(round(rank), d);
    switch GELowRankOpts.method
        case 'svd'
            lowrank = localBuildLowRankCacheSVD(G, r);
        case 'hybrid_residual_subspace'
            lowrank = localBuildLowRankCacheHybridResidualSubspace(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts);
        case 'supervised_joint_operator'
            lowrank = localBuildLowRankCacheSupervisedJointOperator(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts);
        otherwise
            error('aPC:UnsupportedGELowRankMethod', 'Unsupported GELowRankOpts.method: %s', GELowRankOpts.method);
    end
end

function lowrank = localBuildLowRankCacheSVD(G, r)
    [~, Sg, Vg] = svd(G, 'econ');
    lowrank = localFinalizeLowRankCache(G, Vg, diag(Sg), r);
end

function lowrank = localBuildLowRankCacheHybridResidualSubspace(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts)
    if ischar(GELowRankOpts.residual_rank) || (isstring(GELowRankOpts.residual_rank) && isscalar(GELowRankOpts.residual_rank))
        residual_mode = lower(strtrim(char(GELowRankOpts.residual_rank)));
        if strcmp(residual_mode, 'auto')
            lowrank = localBuildLowRankCacheHybridAutoResidualRank(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts);
            return;
        else
            error('aPC:InvalidResidualRankMode', ...
                'GELowRankOpts.residual_rank string mode must be ''auto''.');
        end
    end

    [beta_pilot, residual, F_pilot] = localBuildPilotTrend(Psi, Y, trend_path_idx);
    if isempty(trend_path_idx)
        dF_pilot = zeros(size(U, 1) * size(U, 2), 1);
    else
        dF_pilot = localBuildTrendGradientMatrix(U, basis_info, trend_path_idx);
    end
    G_pilot = localStackedGradientToMatrix(dF_pilot * beta_pilot, size(U, 1), size(U, 2));
    G_res = G - G_pilot;

    [row_weights, weighting_info] = localBuildWeightedGradientRows(G, residual, GELowRankOpts);
    G_main_svd = localApplyRowWeights(G, row_weights);
    G_res_svd = localApplyRowWeights(G_res, row_weights);

    [~, S_main, V_main] = svd(G_main_svd, 'econ');

    r_res = localResolveResidualRank(r, GELowRankOpts);
    r_main = max(r - r_res, 1);
    r_main = min(r_main, size(V_main, 2));

    basis_parts = V_main(:, 1:r_main);
    if r_res > 0
        switch GELowRankOpts.hybrid_residual_mode
            case 'post_project'
                [~, S_res, V_res] = svd(G_res_svd, 'econ');
                if ~isempty(V_res)
                    V_res_proj = V_res - basis_parts * (basis_parts' * V_res);
                    [Q_res, R_res] = qr(V_res_proj, 0);
                    keep_res = min(r_res, sum(abs(diag(R_res)) > 1e-10));
                    if keep_res > 0
                        basis_parts = [basis_parts, Q_res(:, 1:keep_res)];
                    end
                else
                    S_res = [];
                end
            case 'project_then_svd'
                P_perp = eye(size(basis_parts, 1)) - basis_parts * basis_parts';
                G_res_proj = G_res_svd * P_perp;
                [~, S_res, V_res_proj] = svd(G_res_proj, 'econ');
                if ~isempty(V_res_proj)
                    keep_res = min(r_res, size(V_res_proj, 2));
                    if keep_res > 0
                        basis_parts = [basis_parts, V_res_proj(:, 1:keep_res)];
                    end
                else
                    S_res = [];
                end
            otherwise
                error('aPC:UnsupportedHybridResidualMode', ...
                    'Unsupported GELowRankOpts.hybrid_residual_mode: %s.', GELowRankOpts.hybrid_residual_mode);
        end
    else
        S_res = [];
    end

    [Q, ~] = qr(basis_parts, 0);
    if size(Q, 2) < r
        [Q_fill, ~] = qr([basis_parts, null(Q')], 0);
        Q = Q_fill(:, 1:min(r, size(Q_fill, 2)));
    else
        Q = Q(:, 1:r);
    end

    singular_values = [diag(S_main); diag(S_res)];
    lowrank = struct();
    lowrank.rank = size(Q, 2);
    lowrank.V_r = Q;
    lowrank.Z = G * lowrank.V_r;
    lowrank.singular_values = singular_values(:);
    lowrank.main_rank = r_main;
    lowrank.residual_rank = max(size(Q, 2) - r_main, 0);
    lowrank.hybrid_residual_mode = GELowRankOpts.hybrid_residual_mode;
    lowrank.pilot_residual_norm = norm(residual) / max(sqrt(numel(residual)), eps);
    lowrank.weighted_gradient = weighting_info.enabled;
    lowrank.weighting_score = weighting_info.score_name;
    lowrank.weighting_budget = weighting_info.budget;
    if weighting_info.enabled
        lowrank.weighting_energy_ratio = norm(G_main_svd * lowrank.V_r, 'fro')^2 / max(norm(G_main_svd, 'fro')^2, eps);
    else
        lowrank.weighting_energy_ratio = 1.0;
    end
    lowrank.point_weights = row_weights(:);
    lowrank.weighted_point_count = weighting_info.weighted_point_count;
end

function lowrank = localBuildLowRankCacheHybridAutoResidualRank(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts)
    [~, ~, F_pilot] = localBuildPilotTrend(Psi, Y, trend_path_idx);
    grad_weight = GELowRankOpts.operator_grad_weight;
    if ~isscalar(grad_weight) || ~isfinite(grad_weight) || grad_weight < 0
        error('aPC:InvalidHybridAutoGradWeight', ...
            'GELowRankOpts.operator_grad_weight must be a finite scalar >= 0.');
    end

    residual_grid = localResolveResidualRankGrid(r);
    scores = nan(numel(residual_grid), 1);
    value_scores = nan(numel(residual_grid), 1);
    grad_scores = nan(numel(residual_grid), 1);
    candidates = cell(numel(residual_grid), 1);

    for i = 1:numel(residual_grid)
        opts_i = GELowRankOpts;
        opts_i.residual_rank = residual_grid(i);
        candidate = localBuildLowRankCacheHybridResidualSubspace(G, Y, Psi, trend_path_idx, U, basis_info, r, opts_i);
        value_scores(i) = localPilotJointLOOScore(Y, F_pilot, candidate.Z);
        grad_scores(i) = localGradientProjectionLoss(G, candidate.V_r);
        scores(i) = value_scores(i) + grad_weight * grad_scores(i);
        candidate.auto_residual_value_score = value_scores(i);
        candidate.auto_residual_grad_score = grad_scores(i);
        candidate.auto_residual_joint_score = scores(i);
        candidates{i} = candidate;
    end

    [~, best_idx] = min(scores);
    lowrank = candidates{best_idx};
    lowrank.auto_residual_rank_grid = residual_grid(:);
    lowrank.auto_residual_proxy_scores = scores(:);
    lowrank.auto_residual_value_scores = value_scores(:);
    lowrank.auto_residual_grad_scores = grad_scores(:);
    lowrank.auto_residual_selected_rank = residual_grid(best_idx);
    lowrank.auto_residual_selected_index = best_idx;
    lowrank.auto_residual_grad_weight = grad_weight;
end

function lowrank = localBuildLowRankCacheSupervisedJointOperator(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts)
    [~, ~, F_pilot] = localBuildPilotTrend(Psi, Y, trend_path_idx);
    lowrank_svd = localBuildLowRankCacheSVD(G, r);
    lowrank_hybrid = localBuildLowRankCacheHybridResidualSubspace(G, Y, Psi, trend_path_idx, U, basis_info, r, GELowRankOpts);

    eta_grid = GELowRankOpts.operator_eta_grid;
    if ~isnumeric(eta_grid) || isempty(eta_grid) || any(~isfinite(eta_grid(:)))
        error('aPC:InvalidSupervisedOperatorEtaGrid', ...
            'GELowRankOpts.operator_eta_grid must be a finite numeric vector.');
    end
    eta_grid = eta_grid(:)';
    grad_weight = GELowRankOpts.operator_grad_weight;
    if ~isscalar(grad_weight) || ~isfinite(grad_weight) || grad_weight < 0
        error('aPC:InvalidSupervisedOperatorGradWeight', ...
            'GELowRankOpts.operator_grad_weight must be a finite scalar >= 0.');
    end

    V_base = lowrank_svd.V_r;
    V_target = lowrank_hybrid.V_r;
    switch GELowRankOpts.operator_search_method
        case 'grid'
            [lowrank, diag_info] = localSearchSupervisedOperatorGrid(G, Y, F_pilot, V_base, V_target, r, lowrank_svd.singular_values, eta_grid, grad_weight);
        case 'fminbnd'
            [lowrank, diag_info] = localSearchSupervisedOperatorFminbnd(G, Y, F_pilot, V_base, V_target, r, lowrank_svd.singular_values, eta_grid, grad_weight);
        otherwise
            error('aPC:UnsupportedLowRankOperatorSearchMethod', ...
                'Unsupported operator search method: %s.', GELowRankOpts.operator_search_method);
    end
    lowrank.operator_eta_grid = eta_grid(:);
    lowrank.operator_grad_weight = grad_weight;
    lowrank.operator_search_method = GELowRankOpts.operator_search_method;
    lowrank.operator_proxy_scores = diag_info.proxy_scores;
    lowrank.operator_value_scores = diag_info.value_scores;
    lowrank.operator_grad_scores = diag_info.grad_scores;
    lowrank.operator_selected_eta = diag_info.selected_eta;
    lowrank.operator_selected_index = diag_info.selected_index;
    if isfield(diag_info, 'sample_eta')
        lowrank.operator_sample_eta = diag_info.sample_eta;
    end
end

function [lowrank, diag_info] = localSearchSupervisedOperatorGrid(G, Y, F_pilot, V_base, V_target, r, singular_values, eta_grid, grad_weight)
    scores = nan(size(eta_grid));
    value_scores = nan(size(eta_grid));
    grad_scores = nan(size(eta_grid));
    candidates = cell(size(eta_grid));
    for i = 1:numel(eta_grid)
        [candidates{i}, value_scores(i), grad_scores(i), scores(i)] = ...
            localEvaluateOperatorEta(eta_grid(i), G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight);
    end
    [~, best_idx] = min(scores);
    lowrank = candidates{best_idx};
    diag_info = struct();
    diag_info.proxy_scores = scores(:);
    diag_info.value_scores = value_scores(:);
    diag_info.grad_scores = grad_scores(:);
    diag_info.selected_eta = eta_grid(best_idx);
    diag_info.selected_index = best_idx;
end

function [lowrank, diag_info] = localSearchSupervisedOperatorFminbnd(G, Y, F_pilot, V_base, V_target, r, singular_values, eta_grid, grad_weight)
    eta_lb = min(eta_grid);
    eta_ub = max(eta_grid);
    if eta_lb == eta_ub
        [lowrank, value_score, grad_score, score] = localEvaluateOperatorEta(eta_lb, G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight);
        diag_info = struct('proxy_scores', score, 'value_scores', value_score, 'grad_scores', grad_score, 'selected_eta', eta_lb, 'selected_index', 1, 'sample_eta', eta_lb);
        return;
    end

    objective = @(eta) localEvaluateOperatorScoreOnly(eta, G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight);
    opts = optimset('Display', 'off', 'TolX', 1e-2);
    [eta_opt, ~] = fminbnd(objective, eta_lb, eta_ub, opts);
    sample_eta = unique([eta_grid(:); eta_opt]);
    sample_eta = sort(sample_eta(:));

    scores = nan(size(sample_eta));
    value_scores = nan(size(sample_eta));
    grad_scores = nan(size(sample_eta));
    candidates = cell(size(sample_eta));
    for i = 1:numel(sample_eta)
        [candidates{i}, value_scores(i), grad_scores(i), scores(i)] = ...
            localEvaluateOperatorEta(sample_eta(i), G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight);
    end
    [~, best_idx] = min(scores);
    lowrank = candidates{best_idx};
    diag_info = struct();
    diag_info.proxy_scores = scores(:);
    diag_info.value_scores = value_scores(:);
    diag_info.grad_scores = grad_scores(:);
    diag_info.selected_eta = sample_eta(best_idx);
    diag_info.selected_index = best_idx;
    diag_info.sample_eta = sample_eta(:);
end

function score = localEvaluateOperatorScoreOnly(eta, G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight)
    [~, ~, ~, score] = localEvaluateOperatorEta(eta, G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight);
end

function [candidate, value_score, grad_score, joint_score] = localEvaluateOperatorEta(eta, G, Y, F_pilot, V_base, V_target, r, singular_values, grad_weight)
    V_eta = localBlendSubspace(V_base, V_target, eta, r);
    candidate = localFinalizeLowRankCache(G, V_eta, singular_values, r);
    value_score = localPilotJointLOOScore(Y, F_pilot, candidate.Z);
    grad_score = localGradientProjectionLoss(G, candidate.V_r);
    joint_score = value_score + grad_weight * grad_score;
    candidate.operator_eta = eta;
    candidate.operator_value_score = value_score;
    candidate.operator_grad_score = grad_score;
    candidate.operator_joint_score = joint_score;
end

function lowrank = localFinalizeLowRankCache(G, Vg, singular_values, r)
    if isempty(Vg)
        error('aPC:EmptyLowRankBasis', 'Low-rank basis extraction returned an empty basis.');
    end
    r = min(r, size(Vg, 2));
    [Q, ~] = qr(Vg(:, 1:r), 0);
    lowrank = struct();
    lowrank.rank = r;
    lowrank.V_r = Q;
    lowrank.Z = G * lowrank.V_r;
    lowrank.singular_values = singular_values(:);
end

function beta = localLeastSquares(F, Y)
    if rcond(F' * F) < 1e-12
        beta = pinv(F) * Y;
    else
        beta = F \ Y;
    end
end

function score = localPilotJointLOOScore(Y, F_pilot, Z)
    X = [F_pilot, Z];
    X = localRemoveNearConstantColumns(X);
    if isempty(X)
        X = ones(size(Y, 1), 1);
    end

    XtX = X' * X;
    reg = 1e-10 * trace(XtX) / max(size(XtX, 1), 1);
    A = XtX + reg * eye(size(XtX));
    beta = A \ (X' * Y);
    residual = Y - X * beta;

    XAinv = X / A;
    h = sum(XAinv .* X, 2);
    loo_res = residual ./ max(1 - h, 1e-10);
    score = mean(loo_res .^ 2) / max(var(Y, 1), eps);
end

function loss = localGradientProjectionLoss(G, V)
    Gproj = G * V * V';
    loss = norm(G - Gproj, 'fro')^2 / max(norm(G, 'fro')^2, eps);
end

function [beta_pilot, residual, F_pilot] = localBuildPilotTrend(Psi, Y, trend_path_idx)
    if isempty(trend_path_idx)
        F_pilot = ones(size(Y, 1), 1);
    else
        F_pilot = Psi(:, trend_path_idx);
    end
    beta_pilot = localLeastSquares(F_pilot, Y);
    residual = Y - F_pilot * beta_pilot;
end

function V = localBlendSubspace(V_base, V_target, eta, r)
    V_mix = V_base + eta * (V_target - V_base);
    [Q, ~] = qr(V_mix, 0);
    if size(Q, 2) < r
        [Q_fill, ~] = qr([V_mix, null(Q')], 0);
        Q = Q_fill(:, 1:min(r, size(Q_fill, 2)));
    else
        Q = Q(:, 1:r);
    end
    V = Q;
end

function X = localRemoveNearConstantColumns(X)
    if isempty(X)
        return;
    end
    keep = std(X, 0, 1) > 1e-12;
    X = X(:, keep);
end

function [weights, info] = localBuildWeightedGradientRows(G, residual, GELowRankOpts)
    N = size(G, 1);
    weights = ones(N, 1);
    info = struct();
    info.enabled = false;
    info.score_name = GELowRankOpts.weighting_score;
    info.budget = GELowRankOpts.weighting_budget;
    info.weighted_point_count = N;

    if ~GELowRankOpts.weighted_gradient
        return;
    end

    budget = GELowRankOpts.weighting_budget;
    if ~isscalar(budget) || ~isfinite(budget) || budget <= 0
        error('aPC:InvalidWeightedGradientBudget', ...
            'GELowRankOpts.weighting_budget must be a finite scalar > 0.');
    end

    grad_norm = sqrt(sum(G .^ 2, 2));
    switch GELowRankOpts.weighting_score
        case 'gradnorm'
            score = grad_norm;
        case 'residual_gradnorm'
            score = abs(residual(:)) .* grad_norm;
        otherwise
            error('aPC:UnsupportedWeightedGradientScore', ...
                'Unsupported weighted gradient score: %s.', GELowRankOpts.weighting_score);
    end

    if budget <= 1
        keep_count = max(1, ceil(budget * N));
    else
        keep_count = min(N, round(budget));
    end
    [~, order] = sort(score, 'descend');
    keep_ids = order(1:keep_count);
    score_keep = score(keep_ids);
    score_keep = score_keep / max(max(score_keep), eps);
    weights = zeros(N, 1);
    weights(keep_ids) = max(score_keep, 1e-8);

    info.enabled = true;
    info.weighted_point_count = nnz(weights > 0);
end

function G_weighted = localApplyRowWeights(G, weights)
    if isempty(weights)
        G_weighted = G;
        return;
    end
    if numel(weights) ~= size(G, 1)
        error('aPC:InvalidWeightedGradientRows', 'Row weight count must match the number of gradient rows.');
    end
    G_weighted = G .* sqrt(weights(:));
end

function r_res = localResolveResidualRank(r, GELowRankOpts)
    if isempty(GELowRankOpts.residual_rank)
        r_res = min(max(1, floor(r / 3)), max(r - 1, 0));
    else
        r_res = round(GELowRankOpts.residual_rank);
    end
    if ~isscalar(r_res) || ~isfinite(r_res) || r_res < 0
        error('aPC:InvalidResidualSubspaceRank', 'GELowRankOpts.residual_rank must be empty or a finite scalar >= 0.');
    end
    r_res = min(r_res, max(r - 1, 0));
end

function residual_grid = localResolveResidualRankGrid(r)
    residual_grid = 1:max(r - 1, 1);
end

function Xc = localCenterColumns(X)
    if isempty(X)
        Xc = X;
        return;
    end
    Xc = bsxfun(@minus, X, mean(X, 1));
end

function Gmat = localStackedGradientToMatrix(gvec, N, M)
    Gmat = zeros(N, M);
    for a = 1:M
        rows = (a - 1) * N + (1:N);
        Gmat(:, a) = gvec(rows);
    end
end

function dF_r = localProjectGradientTrend(dF, V_r)
    N = size(dF, 1) / size(V_r, 1);
    T = kron(V_r', eye(N));
    dF_r = T * dF;
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
