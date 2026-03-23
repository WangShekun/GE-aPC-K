function [X_red, ReduceModel] = aPC_dimReduction_spca_hsic(U, Y, ncomp)
% ===============================================================
%  aPC_dimReduction_spca_hsic
%
%  自动选择 Primal SPCA / Dual SPCA （HSIC-SPCA）
%  适用于：U 已经 Z-score 标准化（均值 0，标准差 1）
%
%  输入：
%     U     — N×d 输入矩阵（已标准化！）
%     Y     — N×1 或 N×k 标签（未标准化）
%     ncomp — 目标降维维度
%
%  输出：
%     X_red        — 降维后的训练数据（N×ncomp）
%     ReduceModel  — 模型结构（含 transform 回调）
%
% ===============================================================

    [N, d] = size(U);

    % -------------------------
    % 1) U 已标准化，不需要再处理
    % -------------------------
    Uc = U;

    % -------------------------
    % 2) Y 只需中心化
    % -------------------------
    meanY = mean(Y,1);
    Yc = Y - meanY;

    % -------------------------
    % 3) Centering matrix H
    % -------------------------
    H = eye(N) - (1/N)*ones(N);

    % -------------------------
    % 4) Label kernel (linear)
    % -------------------------
    Ky = Yc * Yc.';          % N×N

    % -------------------------
    % 5) AUTO SELECT: PRIMAL / DUAL
    % -------------------------
    if N >= d
        % ====================================
        %       PRIMAL SPCA
        % ====================================
        method = 'spca-primal';

        M = Uc.' * H * Ky * H * Uc;   % d×d
        M = (M + M.') / 2;            % 对称化

        [V_raw, D_raw] = eig(M);
        [eigvals, idx] = sort(diag(D_raw), 'descend');

        W = V_raw(:, idx(1:ncomp));   % d×ncomp

        X_red = Uc * W;

    else
        % ====================================
        %       DUAL SPCA
        % ====================================
        method = 'spca-dual';

        % A = Uc * (H * Ky * H) * Uc.';   % N×N
        A = (Uc * Uc.') * H * Ky * H;   % N×N
        A = (A + A.') / 2;

        [V_raw, D_raw] = eig(A);
        [eigvals, idx] = sort(diag(D_raw), 'descend');

        Alpha = V_raw(:, idx(1:ncomp));   % N×ncomp

        W = Uc.' * Alpha; % d×ncomp

        % 数值稳定性：正交化
        [W, ~] = qr(W, 0);

        X_red = Uc * W;
    end

    % -------------------------
    % 6) 保存模型
    % -------------------------
    ReduceModel.method = method;
    ReduceModel.params.W = W;
    ReduceModel.info.eigvals = eigvals;

    % 注意：U 已标准化，因此 transform 不需要减均值
    ReduceModel.transform = @(U_new)( U_new * W );

end
