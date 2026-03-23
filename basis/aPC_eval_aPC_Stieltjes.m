%% ================= 内部：Stieltjes 三项递推 + 归一化 =================
% H.-P. Wan, W.-X. Ren, and M. D. Todd, 
% “Arbitrary polynomial chaos expansion method for uncertainty quantification
% and global sensitivity analysis in structural dynamics,” 
% Mechanical Systems and Signal Processing, vol. 142, p. 106732, Aug. 2020, 
% doi: 10.1016/j.ymssp.2020.106732.
function [vals, coeffs_polyval] = aPC_eval_aPC_Stieltjes(P, u)
% 构造流程：
%   1) 在样本经验内积下，用三项递推生成 monic 正交多项式 {P_k}
%   2) 计算 ||P_k||^2 = E[P_k^2]，得到正交归一 φ_k = P_k / ||P_k||
%   3) 返回 φ_k(x) 的取值（vals）与 φ_k 的 poly（给 polyval，高次在前）

u = u(:);
N = numel(u);

% --- 保存 monic 正交 P_k 的样本值与系数（升幂） ---
Pvals = zeros(N, P+1);
Cmon  = cell(1, P+1);  % 升幂：常数项在前

% P0 = 1
Pvals(:,1) = 1;
Cmon{1}    = 1;

if P == 0
    % 归一化
    n0 = sqrt(mean(Pvals(:,1).^2));
    if n0 < eps, n0 = 1; end
    vals = Pvals(:,1) / n0;
    coeffs_polyval = {1/n0};           % 给 polyval：高次在前，此处仍然是常数
    return;
end

% P1 = x - α0
alpha0     = mean(u);
Pvals(:,2) = u - alpha0;
Cmon{2}    = [-alpha0, 1];             % 升幂：c0 + c1 x

% 递推生成 Pk
for k = 2:P
    % α_k = <x Pk, Pk> / <Pk,Pk>
    num   = mean(u .* (Pvals(:,k).^2));
    den   = mean(Pvals(:,k).^2);
    alpha = num / max(den, eps);

    % β_k = <Pk,Pk> / <P_{k-1},P_{k-1}>
    den_prev = mean(Pvals(:,k-1).^2);
    beta     = den / max(den_prev, eps);

    % P_{k+1} = (x - α_k) P_k - β_k P_{k-1}
    Pvals(:,k+1) = (u - alpha).*Pvals(:,k) - beta*Pvals(:,k-1);

    % 系数（升幂）递推：
    % (x - α) * poly_k = conv(poly_k, [ -α, 1 ])  —— 注意这里是升幂
    poly1 = conv(Cmon{k}, [-alpha, 1]);
    poly2 = beta * Cmon{k-1};
    % 对齐长度后相减
    L = max(numel(poly1), numel(poly2));
    poly1 = [poly1, zeros(1, L-numel(poly1))];
    poly2 = [poly2, zeros(1, L-numel(poly2))];
    Cmon{k+1} = poly1 - poly2;
end

% -------- 统一归一化：φ_k = P_k / ||P_k|| --------
vals = zeros(N, P+1);
coeffs_polyval = cell(1, P+1);

for k = 0:P
    Pk   = Pvals(:,k+1);
    nk   = sqrt(mean(Pk.^2));
    if nk < eps, nk = 1; end

    % 归一值
    vals(:,k+1) = Pk / nk;

    % 系数转为给 polyval 的“高次在前”
    c_asc = Cmon{k+1} / nk;     % 先按升幂缩放
    c_desc = fliplr(c_asc);     % 再转为降幂（polyval 需要）
    coeffs_polyval{k+1} = c_desc;
end
end