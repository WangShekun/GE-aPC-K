function [K, y_aug, H, lowrank] = pckBuildLowRankAugmentedSystem(U, Y, G, F, dF, theta, corr_type, nugget, rank)
    N = size(U, 1);
    d = size(U, 2);

    if ~isscalar(rank) || ~isfinite(rank) || rank < 1
        error('aPC:InvalidGELowRankRank', 'GELowRankOpts.rank must be a finite scalar >= 1.');
    end
    r = min(round(rank), d);

    [~, Sg, Vg] = svd(G, 'econ');
    V_r = Vg(:, 1:r);
    sing_vals = diag(Sg);
    Z = G * V_r;

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
    H = [F; localProjectGradientTrend(dF, V_r)];

    lowrank = struct();
    lowrank.rank = r;
    lowrank.V_r = V_r;
    lowrank.Z = Z;
    lowrank.singular_values = sing_vals;
end

function dF_r = localProjectGradientTrend(dF, V_r)
    N = size(dF, 1) / size(V_r, 1);
    r = size(V_r, 2);
    T = kron(V_r', eye(N));
    dF_r = T * dF;
end
