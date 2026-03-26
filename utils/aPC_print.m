function aPC_print(results, ModelOpts, data, varargin)
% aPC_print Print a normalized training report for APC/APCK/GEAPCK models.
% Optional modules: "basic", "samples", "errors", "reduction",
% "coefficients", "orthogonality", or "all".

    default_modules = ["basic", "samples", "errors", "reduction"];
    modules = localResolveModules(default_modules, varargin{:});

    width = 84;
    divider = repmat('-', 1, width);
    banner = localBanner('aPC REPORT', width);

    fprintf('\n%s\n', banner);

    if any(modules == "basic")
        localPrintSection('Model Summary', width);
        localPrintKV('Regression method', upper(localGetString(results, 'method', '-')));
        localPrintKV('Model type', upper(localGetString(results, 'ModelType', 'APC')));

        pck_type = localGetString(results, 'PCKType', 'none');
        if ~strcmpi(pck_type, 'none')
            localPrintKV('PCK type', upper(pck_type));
        end

        localPrintKV('Max polynomial order', localFormatScalar(localGetNumeric(ModelOpts, 'Order', '-')));
        localPrintKV('q-norm', localFormatScalar(localGetNumeric(ModelOpts, 'qnorm', '-')));
        localPrintKV('Total PCE terms', localFormatScalar(size(localGetArray(results, 'coefficients'), 1)));

        coeffs = localGetArray(results, 'coefficients');
        active_terms = [];
        if isfield(results, 'PCKModel') && isfield(results.PCKModel, 'trend_idx') && ~isempty(results.PCKModel.trend_idx)
            active_terms = numel(results.PCKModel.trend_idx);
        elseif strcmpi(localGetString(results, 'method', ''), 'lars') && ~isempty(coeffs)
            active_terms = nnz(coeffs);
        end
        if ~isempty(active_terms)
            localPrintKV('Active terms', localFormatScalar(active_terms));
        end

        fprintf('%s\n', divider);
    end

    if any(modules == "samples")
        localPrintSection('Data Summary', width);
        localPrintKV('Training samples', localFormatScalar(localGetNumeric(results, 'N_train', size(localGetArray(data, 'X_train'), 1))));
        localPrintKV('Testing samples', localFormatScalar(localGetNumeric(results, 'N_test', localGetOptionalCount(data, 'X_val'))));
        localPrintKV('Input dimension', localFormatScalar(localResolveInputDim(ModelOpts, data)));
        fprintf('%s\n', divider);
    end

    if any(modules == "errors")
        localPrintSection('Error Summary', width);
        loo = '-';
        if isfield(results, 'optErrorParams') && isfield(results.optErrorParams, 'loo')
            loo = localFormatMetric(results.optErrorParams.loo);
        elseif isfield(results, 'LOO')
            loo = localFormatMetric(results.LOO);
        end
        localPrintKV('LOO error', loo);

        if isfield(results, 'optErrorParams') && isfield(results.optErrorParams, 'ModifiedLoo')
            localPrintKV('Modified LOO', localFormatMetric(results.optErrorParams.ModifiedLoo));
        end

        if isfield(results, 'RelMSE')
            localPrintKV('RelMSE', localFormatMetric(results.RelMSE));
        else
            localPrintKV('RelMSE', '-');
        end
        fprintf('%s\n', divider);
    end

    if any(modules == "reduction") && isfield(ModelOpts, 'ReductionMethod') && ~isempty(ModelOpts.ReductionMethod)
        localPrintSection('Reduction Summary', width);
        localPrintKV('Dimension reduction', upper(ModelOpts.ReductionMethod));
        if isfield(ModelOpts, 'red_dim') && ~isempty(ModelOpts.red_dim)
            localPrintKV('Reduced dimension', localFormatScalar(ModelOpts.red_dim));
        end
        if any(strcmpi(ModelOpts.ReductionMethod, {'kpca', 'skpca_hsic'}))
            localPrintKV('Kernel', aPC_format_kernel_str(ModelOpts));
        end
        fprintf('%s\n', divider);
    end

    if any(modules == "orthogonality")
        localPrintOrthogonality(results, ModelOpts, divider, width);
    end

    if any(modules == "coefficients")
        localPrintSection('PCE Coefficients', width);
        coeffs = localGetArray(results, 'coefficients');
        indices = localResolveIndices(results, ModelOpts);
        fprintf('%-6s %-20s %s\n', 'Idx', 'Multi-Index', 'Coefficient');
        for k = 1:numel(coeffs)
            idx = sprintf('%d,', indices(k, :));
            idx = ['[', idx(1:end-1), ']'];
            fprintf('%6d %-20s % .8e\n', k, idx, coeffs(k));
        end
        fprintf('%s\n', divider);
    end

    fprintf('%s\n', banner);
end

function modules = localResolveModules(default_modules, varargin)
    if isempty(varargin)
        modules = default_modules;
        return;
    end

    requested = string(varargin);
    if any(strcmpi(requested, "all"))
        modules = ["basic", "samples", "errors", "reduction", "orthogonality", "coefficients"];
        return;
    end

    modules = requested;
end

function localPrintSection(title, width)
    fprintf('%s\n', localBanner(title, width));
end

function line = localBanner(title, width)
    inner = sprintf(' %s ', title);
    pad_total = max(0, width - strlength(inner));
    left = floor(pad_total / 2);
    right = pad_total - left;
    line = [repmat('=', 1, left), inner, repmat('=', 1, right)];
end

function localPrintKV(key, val)
    fprintf('%-28s : %s\n', key, string(val));
end

function out = localGetString(S, field, default)
    if isfield(S, field) && ~isempty(S.(field))
        out = string(S.(field));
    else
        out = string(default);
    end
end

function out = localGetNumeric(S, field, default)
    if isfield(S, field) && ~isempty(S.(field))
        out = S.(field);
    else
        out = default;
    end
end

function out = localGetArray(S, field)
    if isfield(S, field) && ~isempty(S.(field))
        out = S.(field);
    else
        out = [];
    end
end

function out = localFormatScalar(val)
    if isequal(val, '-') || isempty(val)
        out = '-';
        return;
    end
    if isnumeric(val)
        if isscalar(val)
            if abs(val - round(val)) < 1e-12
                out = sprintf('%d', round(val));
            else
                out = sprintf('%.4g', val);
            end
        else
            out = mat2str(val);
        end
    else
        out = string(val);
    end
end

function out = localFormatMetric(val)
    if isempty(val)
        out = '-';
    else
        out = sprintf('%.6e', val);
    end
end

function n = localGetOptionalCount(data, field)
    if isfield(data, field) && ~isempty(data.(field))
        n = size(data.(field), 1);
    else
        n = '-';
    end
end

function dim = localResolveInputDim(ModelOpts, data)
    if isfield(ModelOpts, 'num_vars') && ~isempty(ModelOpts.num_vars)
        dim = ModelOpts.num_vars;
    elseif isfield(data, 'X_train') && ~isempty(data.X_train)
        dim = size(data.X_train, 2);
    else
        dim = '-';
    end
end

function Indices = localResolveIndices(results, ModelOpts)
    if isfield(results, 'Indices') && ~isempty(results.Indices)
        Indices = results.Indices;
    else
        Indices = ModelOpts.Indices;
    end
end

function localPrintOrthogonality(results, ModelOpts, divider, width)
    localPrintSection('Orthogonality Diagnostics', width);

    if ~isfield(results, 'univ_vals') || isempty(results.univ_vals)
        fprintf('[WARN] results.univ_vals not found; skip diagnostics.\n');
        fprintf('%s\n', divider);
        return;
    end

    univ_vals = results.univ_vals;
    [N_u, M_u, Q_u] = size(univ_vals);
    Indices = localResolveIndices(results, ModelOpts);
    p_u = Q_u - 1;

    fprintf('Data: N = %d, M = %d, univ basis max order = %d (Q = %d)\n', N_u, M_u, p_u, Q_u);
    fprintf('Indices: size = %d x %d\n', size(Indices, 1), size(Indices, 2));

    p_check = ModelOpts.Order;
    if p_check > p_u
        p_check = p_u;
    end
    nb = p_check + 1;
    I = eye(nb);
    froErr = zeros(1, M_u);
    condEst = zeros(1, M_u);

    for j = 1:M_u
        Phi = squeeze(univ_vals(:, j, 1:nb));
        G = (Phi' * Phi) / size(Phi, 1);
        G = (G + G') / 2;
        froErr(j) = norm(G - I, 'fro');
        ev = sort(real(eig(G)), 'ascend');
        lamMin = ev(1);
        lamMax = ev(end);
        if lamMin <= 0
            condEst(j) = inf;
        else
            condEst(j) = lamMax / lamMin;
        end
        fprintf('Dim %2d | cond ~ %9.3e | ||G-I||_F = %9.3e\n', j, condEst(j), froErr(j));
    end
    fprintf('Summary: max ||G-I||_F = %.3e\n', max(froErr));

    Psi = aPC_PCE_create_Psi(Indices, univ_vals);
    [N, K] = size(Psi);
    fprintf('Psi size = %d x %d (N/K = %.3f)\n', N, K, N / max(K, 1));
    fprintf('rank(Psi) = %g\n', rank(Psi));

    H = Psi' * Psi;
    fprintf('cond(Psi''Psi) = %.3e\n', cond(H));
    fprintf('rcond(Psi''Psi) = %.3e\n', rcond(H));
    try
        fprintf('condest(Psi''Psi) = %.3e\n', condest(H));
    catch
        fprintf('condest(Psi''Psi) = [NA]\n');
    end
    coln = sqrt(sum(Psi.^2, 1));
    fprintf('col-norm: min = %.3e, median = %.3e, max = %.3e\n', min(coln), median(coln), max(coln));
    try
        [Q, ~] = qr(Psi, 0);
        h = sum(Q.^2, 2);
        fprintf('leverage h: max = %.4f, mean = %.4f, std = %.4f\n', max(h), mean(h), std(h));
    catch
        fprintf('leverage h: [WARN] qr(Psi,0) failed.\n');
    end

    fprintf('%s\n', divider);
end
