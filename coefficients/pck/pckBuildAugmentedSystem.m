function [K, y_aug, H] = pckBuildAugmentedSystem(U, Y, G, F, dF, theta, corr_type, nugget)
    % Assemble the GEAPCK block system [Kff Kfg; Kgf Kgg] together with the
    % augmented observation vector and augmented trend matrix.
    N = size(U, 1);
    d = size(U, 2);
    [Kff, Kfg_blocks, Kgg_blocks] = pckEvalKernel(U, U, theta, corr_type);
    if isempty(Kfg_blocks)
        error('aPC:UnsupportedGECorrelation', ...
            'GEAPCK does not support %s correlation because the kernel is not differentiable at zero.', corr_type);
    end

    nobs = N * (d + 1);
    K = zeros(nobs, nobs);
    K(1:N, 1:N) = Kff;

    for a = 1:d
        rows = N + (a - 1) * N + (1:N);
        K(1:N, rows) = Kfg_blocks{a};
        K(rows, 1:N) = -Kfg_blocks{a};
        for b = 1:d
            cols = N + (b - 1) * N + (1:N);
            K(rows, cols) = Kgg_blocks{a, b};
        end
    end

    K(1:nobs + 1:end) = K(1:nobs + 1:end) + nugget;
    y_aug = [Y; reshape(G, N * d, 1)];
    H = [F; dF];
end
