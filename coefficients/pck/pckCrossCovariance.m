function cov_vec = pckCrossCovariance(U_new, U_train, theta, corr_type, use_ge)
    % Cross-covariance between prediction points and training quantities.
    % GEAPCK appends the blocks that couple function values with training gradients.
    [Kff, Kfg_blocks] = pckEvalKernel(U_new, U_train, theta, corr_type);

    if ~use_ge
        cov_vec = Kff;
        return;
    end

    if isempty(Kfg_blocks)
        error('aPC:UnsupportedGECorrelation', 'GE cross-covariance requires differentiable kernel: %s', corr_type);
    end

    cov_vec = [Kff, cat(2, Kfg_blocks{:})];
end
