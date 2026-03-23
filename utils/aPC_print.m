function aPC_print(results, ModelOpts, data, varargin)
% aPC_print - 可选择模块的 Arbitrary PCE 结果打印函数
%
% 可用模块选项：
%   "basic"          - 基本信息
%   "samples"        - 训练/测试信息
%   "errors"         - 误差指标
%   "coefficients"   - 多项式系数
%   "orthogonality"  - 正交性检查
%
% 特殊：
%   "all"            - 打印所有模块
%
% 默认（未指定时）打印：
%   basic + samples + errors

%% ---------- 默认模块 ----------
default_modules = ["basic", "samples", "errors", "orthogonality"];

if nargin < 4
    modules = default_modules;
else
    if any(strcmpi(varargin, "all"))
        modules = ["basic","samples","errors","coefficients","orth"];
    else
        modules = string(varargin);
    end
end
Indices = ModelOpts.Indices;
%% ---------- 打印框架 ----------
line        = repmat('-', 1, 78);
line_double = repmat('=', 1, 78);

fprintf('\n%s\n', line_double);
fprintf('Arbitrary Polynomial Chaos Expansion Report\n');
fprintf('%s\n', line);

%% ============================================================
%% 1) 基本信息（basic）
%% ============================================================
if any(modules == "basic")
    fprintf('%-28s : %s\n', 'Regression method', upper(results.method));
    fprintf('%-28s : %d\n', 'Total PCE terms', size(results.coefficients,1));
    fprintf('%-28s : %d\n', 'Max polynomial order', ModelOpts.Order);
    fprintf('%-28s : %.2f\n', 'q-norm', ModelOpts.qnorm);

    if strcmpi(results.method, 'LARS')
        fprintf('%-28s : %d\n', 'Active terms', nnz(results.coefficients));
    end

    %% ----- 降维信息打印（若存在） -----
    if isfield(ModelOpts,'ReductionMethod') && ~isempty(ModelOpts.ReductionMethod)
        fprintf('%-28s : %s\n', 'Dimension reduction', upper(ModelOpts.ReductionMethod));

        if isfield(ModelOpts,'red_dim') && ~isempty(ModelOpts.red_dim)
            fprintf('%-28s : %d\n', 'Reduced dimension', ModelOpts.red_dim);
        end

        % ---- KPCA 专属 kernel 信息 ----
        if strcmpi(ModelOpts.ReductionMethod,'kpca')
            kernel_str = aPC_format_kernel_str(ModelOpts);
            fprintf('%-28s : %s\n', 'KPCA kernel', kernel_str);
        end
    end


    fprintf('%s\n', line);
end

%% ============================================================
%% 2) 样本信息（samples）
%% ============================================================
if any(modules == "samples")
    if isfield(results,'N_train')
        fprintf('%-28s : %d\n', 'Training samples', results.N_train);
    else
        fprintf('%-28s : %s\n', 'Training samples', '—');
    end

    if isfield(results,'N_test')
        fprintf('%-28s : %d\n', 'Testing samples', results.N_test);
    else
        fprintf('%-28s : %s\n', 'Testing samples', '—');
    end

    fprintf('%-28s : %d\n', 'Input dimension', ModelOpts.num_vars);

    fprintf('%s\n', line);
end

%% ============================================================
%% 3) 误差指标（errors）
%% ============================================================
if any(modules == "errors")
    if isfield(results,'optErrorParams') && isfield(results.optErrorParams,'loo')
        fprintf('%-28s : %.6e\n', 'LOO error', results.optErrorParams.loo);
    else
        fprintf('%-28s : %s\n', 'LOO error', '—');
    end

    if isfield(results,'optErrorParams') && isfield(results.optErrorParams,'ModifiedLoo')
        fprintf('%-28s : %.6e\n', 'Modified LOO', results.optErrorParams.ModifiedLoo);
    else
        fprintf('%-28s : %s\n', 'Modified LOO', '—');
    end

    if isfield(results,'RelMSE')
        fprintf('%-28s : %.6e\n', 'RelMSE', results.RelMSE);
    else
        fprintf('%-28s : %s\n', 'RelMSE', '—');
    end

    fprintf('%s\n', line);
end

%% ============================================================
%% 4) 正交性检查（orthogonality）
%% ============================================================
if any(modules == "orthogonality")

    fprintf('Orthogonality / Conditioning Diagnostics:\n');

    if ~isfield(results,'univ_vals') || isempty(results.univ_vals)
        fprintf('  [WARN] results.univ_vals not found -> skip diagnostics.\n');
        fprintf('%s\n', line);
    else
        univ_vals = results.univ_vals;      % N x M x (P+1)
        [N_u, M_u, Q_u] = size(univ_vals);

        % ---- 取 Indices（优先 results.Indices，其次 ModelOpts.Indices）----
        if isfield(results,'Indices') && ~isempty(results.Indices)
            Indices = results.Indices;
            idx_src = 'results.Indices';
        else
            Indices = ModelOpts.Indices;
            idx_src = 'ModelOpts.Indices';
        end

        % ---- 一维阶数（仅用于打印说明；Psi 构造会用 Indices 指定的阶）----
        p_u = Q_u - 1;

        fprintf('  Data: N = %d, M = %d, univ basis max order = %d (Q = %d)\n', N_u, M_u, p_u, Q_u);
        fprintf('  Indices: size = %d x %d  (from %s)\n', size(Indices,1), size(Indices,2), idx_src);

        %% ---------- (A) Univariate orthogonality sanity check ----------
        % 对每个维度，用 0..min(p_u, ModelOpts.Order) 检查 Gram ≈ I
        p_check = ModelOpts.Order;
        if p_check > p_u, p_check = p_u; end
        nb = p_check + 1;

        fprintf('  [Univariate Gram check] using p = %d (basis size = %d)\n', p_check, nb);

        I = eye(nb);
        froErr  = zeros(1, M_u);
        condEst = zeros(1, M_u);

        for j = 1:M_u
            Phi = squeeze(univ_vals(:, j, 1:nb));     % N x (p+1)
            G   = (Phi' * Phi) / size(Phi,1);
            G   = (G + G')/2;

            froErr(j) = norm(G - I, 'fro');

            ev = eig(G);
            ev = sort(real(ev), 'ascend');
            lamMin = ev(1); lamMax = ev(end);
            if lamMin <= 0, condEst(j) = inf; else, condEst(j) = lamMax/lamMin; end

            fprintf('    Dim %2d | cond ~ %9.3e | ||G-I||_F = %9.3e\n', j, condEst(j), froErr(j));
        end
        fprintf('    Summary: max ||G-I||_F = %.3e\n', max(froErr));

        %% ---------- (B) Multivariate design-matrix diagnostics (Psi) ----------
        % 由 Indices + univ_vals 构造 Psi
        try
            Psi = aPC_PCE_create_Psi(Indices, univ_vals);   % N x K
        catch ME
            fprintf('  [Psi] FAILED to create Psi: %s\n', ME.message);
            fprintf('%s\n', line);
            return;
        end

        [N, K] = size(Psi);
        fprintf('  [Psi diagnostics] Psi created: size = %d x %d (N/K = %.3f)\n', N, K, N/max(K,1));

        % rank (数值秩)
        try
            rk = rank(Psi);
        catch
            rk = NaN;
        end
        fprintf('    rank(Psi) = %g (max = %d)\n', rk, min(N,K));

        % Normal matrix conditioning
        H = Psi' * Psi;

        cPsi = cond(H);
        fprintf('    cond(Psi''Psi) = %.3e\n', cPsi)

        rc = rcond(H);
        fprintf('    rcond(Psi''Psi) = %.3e\n', rc);
        

        try
            ce = condest(H);
            fprintf('    condest(Psi''Psi) ~ %.3e\n', ce);
        catch
            fprintf('    condest(Psi''Psi) ~ [NA]\n');
        end

        % Optional: report diag dominance / column scaling
        coln = sqrt(sum(Psi.^2, 1));
        fprintf('    col-norm: min = %.3e, median = %.3e, max = %.3e\n', min(coln), median(coln), max(coln));

        % Leverage via QR: h_i = ||Q(i,:)||^2
        try
            [Q, ~] = qr(Psi, 0);
            h = sum(Q.^2, 2);
            fprintf('    leverage h: max = %.4f, mean = %.4f, std = %.4f\n', max(h), mean(h), std(h));
        catch
            fprintf('    leverage h: [WARN] qr(Psi,0) failed (size/memory).\n');
        end

        fprintf('%s\n', line);
    end
end


%% ============================================================
%% 5) 系数（coefficients）
%% ============================================================
if any(modules == "coefficients")
    fprintf('PCE Coefficients:\n');
    fprintf('%-6s %-20s %s\n', 'Idx', 'Multi-Index', 'Coefficient');

    num_terms = length(results.coefficients);

    for k = 1:num_terms
        idx = sprintf('%d,', Indices(k,:));
        idx = ['[', idx(1:end-1), ']'];
        fprintf('%6d %-20s % .8e\n', k, idx, results.coefficients(k));
    end

    fprintf('%s\n', line);
end

%% ---------- 结束：总是打印双横线 ----------
fprintf('%s\n', line_double);

end
