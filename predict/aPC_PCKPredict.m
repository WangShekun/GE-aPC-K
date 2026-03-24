function y_pred = aPC_PCKPredict(Psi_new, U_new, PCKModel)

    % Evaluate the trend first, then apply the Kriging correction from the stored
    % APCK/GEAPCK model.

    trend_idx = PCKModel.trend_idx;
    Fnew = Psi_new(:, trend_idx);

    U_train = PCKModel.U_train;
    theta = PCKModel.theta;
    use_ge = isfield(PCKModel, 'use_ge') && PCKModel.use_ge;
    cov_vec = pckCrossCovariance(U_new, U_train, theta, PCKModel.corr, use_ge);

    y_pred = Fnew * PCKModel.beta + cov_vec * PCKModel.alpha;
end
