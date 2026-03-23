function orthErr = aPC_check_orthogonality(univ_vals)
% aPC_check_orthogonality
% -------------------------------
% 计算 aPC 一维多项式族的正交性误差（经验 Gram 矩阵）
%
% 输入:
%   univ_vals : N × (p+1) 的矩阵，包含 0..p 阶多项式的评估值
%
% 输出:
%   orthErr : Frobenius 范数误差 ||G - I||_F
% -------------------------------

    [N, P1] = size(univ_vals);   % P1 = p+1

    % 经验 Gram 矩阵 G = (1/N) * Psi^T Psi
    G = (univ_vals' * univ_vals) / N;

    % 单位矩阵
    I = eye(P1);

    % Frobenius norm
    orthErr = norm(G - I, 'fro');

    % 友好打印
    % fprintf('   Orthogonality error (||G - I||_F) = %.3e  (P = %d)\n', ...
    %     orthErr, P1-1);

end
