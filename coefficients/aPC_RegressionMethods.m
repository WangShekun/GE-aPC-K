function results = aPC_RegressionMethods(X, Y, method)
% PC_RegressionMethods  统一回归入口（各子函数均返回 results 结构体）
% 支持：'OLS' | 'LARS' | 'Lasso' | 'Ridge'
% 返回的 results 至少包含：
%   .coefficients, .LOO, .normEmpErr, .method
%
% 用法示例：
%   res  = PC_RegressionMethods(Psi, y, 'OLS');
%   beta = res.coefficients;

    % 确保 Y 为列向量
    if size(Y,2) > 1
        Y = Y(:);
    end

    % 默认方法
    if nargin < 3 || isempty(method)
        method = 'OLS';
    end

    switch lower(strtrim(method))
        case {'ols'}
            results = aPC_OLS(X, Y);

        case {'lars'}
            results = aPC_LARS(X, Y);

        case 'lasso'
            results = aPC_Lasso(X, Y);

        case 'ridge'
            results = aPC_Ridge(X, Y);
            
        case 'omp'
            results = aPC_OMP(X, Y);

        otherwise
            error('Unsupported regression method: %s', method);
    end
end
