function [Psi, poly_coeffs, univ_vals] = aPC_PCE_build_Psi(ModelOpts, data)
% 对齐 UQLab：X->U，评 0..P 阶一维基，再按多索引装配 Psi

% 只保留非常数维度的多索引
IndicesFull = ModelOpts.Indices;

% 最大阶数
Pmax = max(IndicesFull, [], 'all');

% 2) 对每一维一次性评估 0..P 阶一维 aPC，并拿到多项式系数
[univ_vals, poly_coeffs] = aPC_eval_univ_basis(data.U, Pmax, ModelOpts);

% 3) 构造多维基函数矩阵 Psi
Psi = aPC_PCE_create_Psi(IndicesFull, univ_vals);
end
