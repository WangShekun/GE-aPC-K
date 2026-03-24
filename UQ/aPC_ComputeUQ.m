function UQ = aPC_ComputeUQ(results, Data)
% aPC_ComputeUQ
% --------------------------------------------------------
% 基于 aPC-PCE 系数（在正交归一基下）计算全局统计量：
%   - 均值  E[Y]
%   - 方差 Var[Y]
%   - 标准差 Std[Y]
%
% 要求：
%   results.coefficients : K×1 PCE 系数向量 c_k
%   results.Indices      : K×M 多指数矩阵 alpha
%
% 在正交归一基 {φ_k} 下：
%   E[Y]   ≈ c_0
%   Var[Y] ≈ Σ_{k≠0} c_k^2
% --------------------------------------------------------
UQ.hasStats = false;

if isfield(results, 'ModelType') && ~strcmpi(results.ModelType, 'apc')
    error('aPC:UQOnlyForAPC', ...
        'aPC_ComputeUQ currently supports APC only. Current ModelType = %s.', results.ModelType);
end

c       = results.coefficients(:);
Indices = results.Indices;   % K×M

% ---- 找常数基（alpha 全 0） ----
isConst = all(Indices == 0, 2);
if ~any(isConst)
    error('No constant basis found (row with all-zero multi-index).');
end
if nnz(isConst) > 1
    warning('Multiple constant bases found, using the first one.');
end

idx0 = find(isConst, 1, 'first');
c0   = c(idx0);

% ---- 均值 ----
meanY = c0;

% ---- 方差：排除常数项，直接 ∑ c_k^2 ----
maskVar      = true(size(c));
maskVar(idx0)= false;
c_eff        = c(maskVar);
VarY         = sum(c_eff.^2);

% ---- 输出 ----
UQ.mean = meanY;
UQ.var  = VarY;
UQ.std  = sqrt(VarY);

% 方便后续 Sobol 使用
UQ.idx0   = idx0;
UQ.VarTot = VarY;

if nargin >= 2 && isfield(Data,'Y_pred') && ~isempty(Data.Y_pred)
    y = Data.Y_pred(:);
    if numel(y) < 10
        warning('Too few samples in Data.Y_pred (<10), statistics may be unreliable.');
    end

    % 95% 置信区间
    UQ.CI95 = quantile(y, [0.025 0.975]);

    % 偏度 & 峰度
    UQ.skew = skewness(y);
    UQ.kurt = kurtosis(y);

    % VaR95（右尾 95% 分位）和 CVaR95（该尾部的均值）
    alpha    = 0.95;
    UQ.VaR95 = quantile(y, alpha);

    tail = y(y >= UQ.VaR95);
    if isempty(tail)
        UQ.CVaR95 = UQ.VaR95;
    else
        UQ.CVaR95 = mean(tail);
    end

    UQ.hasStats = true;
end

end
