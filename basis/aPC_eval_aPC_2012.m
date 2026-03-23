function [VALUE, poly_coeffs] = aPC_eval_aPC_2012(ORDER, X)
% aPC_eval_aPC:
%   构造并评估一维 aPC 正交归一多项式 φ_0,...,φ_ORDER
%   使用样本 raw moments 构造 Hankel 系统，先得到正交多项式 P_k，
%   再对每一阶做 L2 归一化，使得在经验测度下
%       E[φ_k(X) φ_j(X)] ≈ δ_{kj}
%
% 输入：
%   ORDER : 最大阶数
%   X     : N×1 或 1×N 向量（这里通常是已经标准化后的 U）
%
% 输出：
%   VALUE       : N × (ORDER+1)，第 j 列为 φ_{j-1}(X)
%   poly_coeffs : 1×(ORDER+1) cell，
%                 第 k+1 个元素是 φ_k(x) 的多项式系数（polyval 约定：最高次在前）

% --- 保证列向量 ---
X = X(:);
N = numel(X);

VALUE       = zeros(N, ORDER+1);
poly_coeffs = cell(1, ORDER+1);

% ---- 0 阶：φ_0(x) = 1，本身就是正交归一 ----
VALUE(:,1)     = 1;
poly_coeffs{1} = 1;   % p(x) = 1

if ORDER == 0
    return;
end

% --- 计算 raw moments m_0 ... m_{2*ORDER-1} ---
max_m = 2*ORDER - 1;
m = zeros(1, max_m+1);
for k = 0:max_m
    m(k+1) = mean(X.^k);
end

% ---- 对每个阶 k ≥ 1 构造正交 + 归一一维基 φ_k ----
for k = 1:ORDER
    % 1) 构造 Hankel 线性系统 A * p = b
    % A 为 (k+1)×(k+1)：
    %   - 前 k 行：正交条件 ∑_ℓ p_ℓ m_{r-1+ℓ} = 0, r=1..k
    %   - 最后一行：约束最高次项 p_k = 1
    A = zeros(k+1, k+1);

    % 正交条件行
    for r = 1:k
        % 对应 x^{r-1} 的原始矩：m_{r-1} ... m_{r-1+k}
        A(r,:) = m(r : r+k);
    end

    % 最高次项约束：p_k = 1
    A(end,end) = 1;

    % RHS：前 k 行为 0，最后行为 1
    b = zeros(k+1,1);
    b(end) = 1;

    % 2) 求解多项式系数 p_0,...,p_k
    rA = rcond(A);
    if rA > 1e-12
        p = A \ b;
    else
        % 矩阵病态时使用伪逆，提高鲁棒性
        p = pinv(A) * b;
    end

    % 3) 转成 polyval 约定形式：最高次在前
    p_poly = flipud(p);          % 列 → 高阶在前

    % 4) 在当前样本 X 上评估“未归一”的 P_k(X)
    Pk = polyval(p_poly, X);

    % 5) 估计范数 ||P_k||^2 ≈ E[P_k^2]
    norm2 = mean(Pk.^2);

    if norm2 < eps
        % 极度病态时避免被 0 除，保持原尺度
        scale = 1.0;
    else
        scale = sqrt(norm2);
    end

    % 6) 得到“正交归一”的基函数 φ_k(x)
    phi_k        = Pk / scale;
    p_poly_norm  = p_poly / scale;

    % 7) 存储：φ_k(X) 与其多项式系数
    VALUE(:, k+1)    = phi_k;
    poly_coeffs{k+1} = p_poly_norm.';   % 行向量，便于 polyval
end


end
