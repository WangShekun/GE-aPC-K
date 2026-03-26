function results = aPC_Lasso(X, Y)
% PC_Lasso  Lasso 回归（与 LARS/OLS 一致返回 results 结构体）
% - 使用 10 折交叉验证选择 LambdaMinMSE
% - 不标准化、不拟合截距（PCE 常数基负责常数项）
% - LOO 采用未正则化的 (X'X)^{-1} 近似（与 UQLab 习惯一致）

    if size(Y,2) > 1, Y = Y(:); end

    % 10-fold CV 选 lambda（最小MSE）
    [B, FitInfo] = lasso( ...
        X, Y, ...
        'CV',          10, ...
        'Standardize', false, ...   % 不标准化
        'Intercept',   false, ...   % 不拟合截距
        'RelTol',      1e-6, ...
        'MaxIter',     1e5 );

    idx  = FitInfo.IndexMinMSE;          % 若想更稀疏可改为 Index1SE
    beta = B(:, idx);

    % 评估：modified LOO（用未惩罚的 Hat 矩阵近似）
    XtX = X' * X;
    if rcond(XtX) > 1e-12
        Minv = XtX \ eye(size(XtX));
    else
        Minv = pinv(XtX);
    end
    [LOO, normEmpErr, opt] = aPC_PCE_loo_error(X, Minv, Y, beta, 1);

    % 组织结果
    results = struct();
    results.coefficients   = beta;
    results.LOO            = LOO;
    results.normEmpErr     = normEmpErr;
    results.RegressionMethod = 'LASSO';
    results.lambda         = FitInfo.Lambda(idx);
    results.IndexMinMSE    = idx;
    results.MSEMin         = FitInfo.MSE(FitInfo.IndexMinMSE);
    results.MSE1SE         = FitInfo.MSE(FitInfo.Index1SE);
    results.nNonZero       = nnz(beta);
    results.optErrorParams = opt;     % 含 T 修正等
end
