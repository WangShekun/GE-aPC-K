function y_pred = aPC_PCKPredictLowRank(Psi_new, U_new, PCKModel)

    trend_idx = PCKModel.trend_idx;
    Fnew = Psi_new(:, trend_idx);

    U_train = PCKModel.U_train;
    theta = PCKModel.theta;
    V_r = PCKModel.lowrank.V_r;
    cov_vec = localLowRankCrossCovariance(U_new, U_train, theta, PCKModel.corr, V_r);

    y_pred = Fnew * PCKModel.beta + cov_vec * PCKModel.alpha;
end

function cov_vec = localLowRankCrossCovariance(U_new, U_train, theta, corr_type, V_r)
    [Kff, Kfg_modes] = pckEvalProjectedKernel(U_new, U_train, theta, corr_type, V_r);

    cov_vec = [Kff, cat(2, Kfg_modes{:})];
end
