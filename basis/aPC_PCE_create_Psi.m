function Psi = aPC_PCE_create_Psi(Indices, univ_vals)
% 根据多索引 Indices 与一维基值 univ_vals 组合多维基 Ψ
%
% 输入：
%   Indices   : K×M，多指数
%   univ_vals : N×M×(P+1)，univ_vals(:,j,k+1) = 第 j 维的 ψ_k(U_j)
%
% 输出：
%   Psi       : N×K 设计矩阵

    [N, M, ~] = size(univ_vals);
    K = size(Indices,1);

    if size(Indices,2) ~= M
        error('Indices dim mismatch.');
    end

    Psi = ones(N,K);

    for j = 1:M
        deg = Indices(:,j); 
        mask = deg > 0;
        if any(mask)
            cols = find(mask);
            tmp  = zeros(N, numel(cols));
            for c = 1:numel(cols)
                k = cols(c); 
                d = deg(k);
                tmp(:,c) = univ_vals(:, j, d+1);
            end
            Psi(:,mask) = Psi(:,mask) .* tmp;
        end
    end
end
