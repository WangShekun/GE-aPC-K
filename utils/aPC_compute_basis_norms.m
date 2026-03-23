function gamma = aPC_compute_basis_norms(results)
% aPC_compute_basis_norms
% --------------------------------------------------------
%  依据训练样本上的一维基函数值 univ_vals 和多指数 Indices
%  近似计算每个多元基函数 ψ_k 的二阶矩：
%
%    gamma_k ≈ (1/N) Σ_n ψ_k(x^(n))^2
%
%  其中 ψ_k(x) = ∏_j φ_{alpha_{k,j}}(U_j)
% --------------------------------------------------------

Indices   = results.Indices;      % K×M
univ_vals = results.univ_vals;   % N×M×(P+1)

[K, M] = size(Indices);
[N, M_check, P1] = size(univ_vals);

if M_check ~= M
    error('Dimension mismatch: Indices has %d dims but univ_vals has %d.', M, M_check);
end

gamma = zeros(K,1);

for k = 1:K
    alpha_k = Indices(k,:);     % 1×M
    psi_k   = ones(N,1);        % N×1

    for j = 1:M
        deg = alpha_k(j);
        if deg > 0
            % 对应维度 j 的 deg 阶一维基函数值
            phi_j = univ_vals(:, j, deg+1);   % N×1
            psi_k = psi_k .* phi_j;
        else
            % deg = 0 ⇒ φ_0 = 1，无需操作
        end
    end

    % 二阶矩（在经验分布下）
    gamma(k) = mean(psi_k.^2);
end

end
