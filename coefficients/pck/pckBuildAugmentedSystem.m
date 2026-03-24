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

    Kfg = cat(2, Kfg_blocks{:});
    Kgf = -cat(1, Kfg_blocks{:});
    Kgg = cell2mat(Kgg_blocks);

    K = [Kff, Kfg; Kgf, Kgg] + nugget * eye(N * (d + 1));
    y_aug = [Y; reshape(G, N * d, 1)];
    H = [F; dF];
end
