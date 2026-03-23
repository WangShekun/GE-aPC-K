function [VALUE, poly_coeffs] = aPC_eval_aPC_2018(ORDER, X)
% aPC_eval_aPC_2018
% --------------------------------------------------------
% 基于 2018 版 aPC (Degen et al.) 的正交化构造：
%
%   1. 使用矩形式 (moment matrix) 构造 P^{(k)} —— 正交但未归一；
%   2. 使用论文式 (12):
%          kappa_k = p_k^{(k)}
%      （即最高次项系数作为该多项式的范数因子）
%   3. 使用式 (14) 归一化：
%          phi_k(x) = P^{(k)}(x) / sqrt(kappa_k)
%
% 输出：
%   VALUE       : N × (ORDER+1)，每列为 φ_0,..., φ_ORDER
%   poly_coeffs : 每阶 φ_k 的“归一化后”的系数（用于预测）
% --------------------------------------------------------

X = X(:);          % N×1
N = numel(X);

VALUE       = zeros(N, ORDER+1);
poly_coeffs = cell(1, ORDER+1);

% --------------------------------------------------------
% k = 0 : φ0(x) = 1
% --------------------------------------------------------
VALUE(:,1)      = 1;
poly_coeffs{1}  = 1;   % p0(x)=1

if ORDER == 0
    return;
end

% === 计算原始原点矩 (raw moments) m_0 ... m_{2P} ===
max_m = 2*ORDER;
m = zeros(1, max_m+1);
for k = 0:max_m
    m(k+1) = mean(X.^k);
end

% --------------------------------------------------------
% 逐阶构造 P^{(k)}（正交但未归一）并做归一化
% --------------------------------------------------------
for k = 1:ORDER
    
    % Hankel 系统维度： (k+1)x(k+1)
    A = zeros(k+1, k+1);
    
    % 前 k 行：正交条件 <P^{(k)}, x^j> = 0
    % 即：sum_i p_i * m_{i+j} = 0
    for row = 1:k
        A(row, :) = m(row : row+k);
    end
    
    % 辅助条件（2012 有；2018 仍保留结构但解释不同）
    % 最高次项 p_k^{(k)} 可以自由求，最后归一化处理。
    A(end,end) = 1;  % 保持可解，但不再用来强制归一化
    
    b = zeros(k+1,1);
    b(end) = 1;
    
    % --- 求解未归一化系数 p^{(k)} ---
    rA = rcond(A);
    if rA > 1e-12
        p = A \ b;
    else
        p = pinv(A) * b;
    end
    
    % ---- 将阶从低到高翻转，用于 polyval ----
    p_poly = flipud(p);   % 列向量，高阶在前
    
    % ---- 论文式 (12)：kappa_k = p_k^{(k)} ----
    % p_poly(1) 就是最高次项 p_k^{(k)}
    kappa_k = p_poly(1);
    
    % ---- 论文式 (14)：归一化 φ_k(x) = P^{(k)}(x) / sqrt(kappa_k) ----
    scale       = sqrt(kappa_k);
    p_poly_norm = p_poly / scale;           % 归一化后的多项式系数
    phi_k       = polyval(p_poly_norm, X);  % 归一化后的多项式值
    
    % ---- 存储 ----
    VALUE(:,k+1)       = phi_k;
    poly_coeffs{k+1}   = p_poly_norm.';     % 行向量形式
end

end
