function results = aPC_OLS(Psi, Y)
% PC_OLS  OLS回归（UQLab风格：返回results结构体）
% 用法：
%   results = PC_OLS(Psi, Y);                 % 默认 modified LOO
%
% 返回：
%   results.coefficients   : P×1
%   results.LOO            : 原始LOO
%   results.modifiedLoo    : 修正LOO
%   results.normEmpErr     : 归一化经验误差
%   results.optErrorParams : 附加信息（含 T 修正因子等）
%   results.coeff_array    : 迭代过程中所有的系数轨迹

    % 设置默认值
    modified_loo = true;  % 默认启用 modified LOO
    COVFLAG = false;  % 默认不使用协方差

    % 如果 Y 是多列，转换为列向量
    if size(Y, 2) > 1
        Y = Y(:);
    end

    % ---- 检查是否有协方差矩阵 CY ----
    if COVFLAG
        CY    = options.CY;
        CYinv = CY \ eye(size(CY));
        L     = chol(CYinv);
        Psi   = L * Psi;
        Y     = L * Y;
    end

    % ---- 解 Psi'Psi a = Psi'Y（与UQLab相同的 rcond 分支）----
    PsiTPsi = Psi.' * Psi;
    PsiTY   = Psi.' * Y;

    if rcond(PsiTPsi) > 1e-12
        % faster
        coefficients = PsiTPsi \ PsiTY;
        M            = PsiTPsi \ speye(size(PsiTPsi));
    else
        % stabler
        coefficients = pinv(PsiTPsi) * PsiTY;
        M            = pinv(PsiTPsi);
    end

    % 计算（modified）LOO 与误差
    [modifiedLoo,modifiednormEmpErr,opt_results] = aPC_PCE_loo_error(Psi, M, Y, coefficients, modified_loo);

    % 打包结果
    results = struct();
    results.coefficients   = coefficients;
    results.LOO    = modifiedLoo;  % 修正 LOO
    results.normEmpErr     = modifiednormEmpErr;   % 归一化经验误差
    results.optErrorParams = opt_results;              % 包含T修正因子等
    results.coeff_array    = coefficients(:).';        % 由于只是一个OLS，不需要多个迭代，直接把系数作为迭代轨迹
    results.method = 'OLS';
end
