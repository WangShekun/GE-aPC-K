function [K, y_aug, H, lowrank] = pckBuildLowRankAugmentedSystem(U, Y, G, F, dF, theta, corr_type, nugget, rank, lowrank_cache, dF_r)
    N = size(U, 1);
    d = size(U, 2);

    if nargin < 10 || isempty(lowrank_cache)
        if ~isscalar(rank) || ~isfinite(rank) || rank < 1
            error('aPC:InvalidGELowRankRank', 'GELowRankOpts.rank must be a finite scalar >= 1.');
        end
        r = min(round(rank), d);
        [~, Sg, Vg] = svd(G, 'econ');
        lowrank = struct();
        lowrank.rank = r;
        lowrank.V_r = Vg(:, 1:r);
        lowrank.Z = G * lowrank.V_r;
        lowrank.singular_values = diag(Sg);
    else
        lowrank = localNormalizeLowRankCache(lowrank_cache, rank, d);
        r = lowrank.rank;
    end

    V_r = lowrank.V_r;
    Z = lowrank.Z;

    [Kff, Kfg_modes, Kgg_modes] = pckEvalProjectedKernel(U, U, theta, corr_type, V_r);

    nobs = N * (r + 1);
    K = zeros(nobs, nobs);
    K(1:N, 1:N) = Kff;

    for k = 1:r
        rows = N + (k - 1) * N + (1:N);
        K(1:N, rows) = Kfg_modes{k};
        K(rows, 1:N) = -Kfg_modes{k};
        for l = 1:r
            cols = N + (l - 1) * N + (1:N);
            K(rows, cols) = Kgg_modes{k, l};
        end
    end

    K(1:nobs + 1:end) = K(1:nobs + 1:end) + nugget;
    y_aug = [Y; reshape(Z, N * r, 1)];
    if nargin < 11 || isempty(dF_r)
        dF_r = localProjectGradientTrend(dF, V_r);
    end
    H = [F; dF_r];
end

function dF_r = localProjectGradientTrend(dF, V_r)
    N = size(dF, 1) / size(V_r, 1);
    r = size(V_r, 2);
    T = kron(V_r', eye(N));
    dF_r = T * dF;
end

function lowrank = localNormalizeLowRankCache(lowrank_cache, rank, d)
    if ~isscalar(rank) || ~isfinite(rank) || rank < 1
        error('aPC:InvalidGELowRankRank', 'GELowRankOpts.rank must be a finite scalar >= 1.');
    end
    r = min(round(rank), d);
    if ~isstruct(lowrank_cache) || ~isfield(lowrank_cache, 'V_r') || ~isfield(lowrank_cache, 'Z')
        error('aPC:InvalidLowRankCache', 'Low-rank cache must contain V_r and Z.');
    end
    if size(lowrank_cache.V_r, 2) < r || size(lowrank_cache.Z, 2) < r
        error('aPC:InvalidLowRankCache', 'Low-rank cache does not contain enough rank components.');
    end
    lowrank = lowrank_cache;
    lowrank.rank = r;
    lowrank.V_r = lowrank_cache.V_r(:, 1:r);
    lowrank.Z = lowrank_cache.Z(:, 1:r);
    if isfield(lowrank_cache, 'singular_values') && ~isempty(lowrank_cache.singular_values)
        lowrank.singular_values = lowrank_cache.singular_values;
    else
        lowrank.singular_values = zeros(min(size(lowrank_cache.Z, 1), size(lowrank_cache.V_r, 1)), 1);
    end
end
