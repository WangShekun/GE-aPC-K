function [univ_vals, poly_coeffs] = aPC_eval_univ_basis(U, MaxDegrees, ModelOpts)
% aPC_eval_univ_basis
% --------------------------------------------------------
% 输入:
%   U : N×M 的规范化输入
%   BasisParameters.MaxDegrees = P
%   ModelOpts.aPC_Method = '2012' | '2018' | 其他
%
% 输出：
%   univ_vals  : N×M×(P+1)
%   poly_coeffs: 1×M cell，每个维度 1×(P+1) 的多项式系数
% --------------------------------------------------------

P = MaxDegrees;
[N, M] = size(U);

univ_vals   = zeros(N, M, P+1);
poly_coeffs = cell(1, M);

% 根据用户选择的方法，调用不同的一维构造函数
method = lower(ModelOpts.aPC_Method);

for j = 1:M
    uj = U(:, j);

    switch method
        case '2012'
            [vals_j, coeffs_j] = aPC_eval_aPC_2012(P, uj);

        case '2018'
            [vals_j, coeffs_j] = aPC_eval_aPC_2018(P, uj);

        case 'stieltjes'
            [vals_j, coeffs_j] = aPC_eval_aPC_Stieltjes(P, uj);

        otherwise
            error('Unknown aPC method "%s".', ModelOpts.aPC_Method);
    end

    univ_vals(:, j, :) = vals_j;
    poly_coeffs{j}     = coeffs_j;
end

end
