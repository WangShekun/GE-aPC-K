function results = aPC_Ridge(X, Y, lambda)
% PC_Ridge  岭回归（与 LARS/OLS 一致返回 results 结构体）
% - 若未给 lambda，则用 5 折 CV 在 logspace(-6,2,25) 上选最优
% - 不标准化、不拟合截距
% - LOO 采用未正则化的 (X'X)^{-1} 近似（与 UQLab 习惯一致）

    if size(Y,2) > 1, Y = Y(:); end
    [N,P] = size(X);

    % —— 选择/确定 lambda —— 
    if nargin < 3 || isempty(lambda)
        lambda_grid = logspace(-6, 2, 25);
        K = 5;
        cv_idx = crossvalind('Kfold', N, K);

        mse_cv = zeros(numel(lambda_grid), 1);
        for li = 1:numel(lambda_grid)
            lam = lambda_grid(li);
            se_acc = zeros(K,1);
            for k = 1:K
                val = (cv_idx == k);
                trn = ~val;

                Xtr = X(trn, :);   Ytr = Y(trn);
                Xvl = X(val, :);   Yvl = Y(val);

                A = (Xtr' * Xtr) + lam * eye(P);
                b = (Xtr' * Ytr);
                if rcond(A) > 1e-12
                    beta = A \ b;
                else
                    beta = pinv(A) * b;
                end
                yhat = Xvl * beta;
                se_acc(k) = mean((Yvl - yhat).^2);
            end
            mse_cv(li) = mean(se_acc);
        end
        [~, best_idx] = min(mse_cv);
        lambda = lambda_grid(best_idx);
    end

    % —— 最终拟合 —— 
    A = (X' * X) + lambda * eye(P);
    b = (X' * Y);
    if rcond(A) > 1e-12
        beta = A \ b;
    else
        beta = pinv(A) * b;
    end

    % —— 评估：modified LOO（未正则化 Hat 矩阵近似）——
    XtX = X' * X;
    if rcond(XtX) > 1e-12
        Minv = XtX \ eye(size(XtX));
    else
        Minv = pinv(XtX);
    end
    [LOO, normEmpErr, opt] = aPC_PCE_loo_error(X, Minv, Y, beta, 1);

    % —— 组织结果 —— 
    results = struct();
    results.coefficients   = beta;
    results.LOO            = LOO;
    results.normEmpErr     = normEmpErr;
    results.RegressionMethod = 'RIDGE';
    results.lambda         = lambda;
    results.nNonZero       = nnz(beta);   % Ridge通常全非零，仅作记录
    results.optErrorParams = opt;
end
