function [X_red, ReduceModel] = aPC_dimReduction_spca_hsic(U, Y, ncomp)
% ===============================================================
%  aPC_dimReduction_spca_hsic
%
%  HSIC-SPCA (primal, solved by SVD)
%  适用于：U 已经 Z-score 标准化（均值 0，标准差 1）
%
%  输入：
%     U     — N×d 输入矩阵（已标准化）
%     Y     — N×1 或 N×k 标签/响应（未标准化）
%     ncomp — 目标降维维度
%
%  输出：
%     X_red        — 降维后的训练数据（N×r_use）
%     ReduceModel  — 模型结构（含 transform 回调）
%
%  原理：
%     max_W  tr(W' * U' * H * Ky * H * U * W)
%     s.t.   W'W = I
%
%     其中：
%         H  = I - (1/N) * 11'
%         Ky = Yc * Yc'   （线性标签核）
%
%  数值实现：
%     令 B = H * Ky * H = Delta * Delta'
%     则
%         M = U' * B * U = (Delta' * U)' * (Delta' * U)
%     因此可通过对
%         Psi = Delta' * U
%     做经济型 SVD 来获得投影方向。
%
%  说明：
%     1) 本函数仅实现标准 primal HSIC-SPCA；
%     2) 若有效监督维数小于 ncomp，则自动截断；
%     3) transform 输入的 U_new 需要已按训练集方式标准化。
%
% ===============================================================

    [N, d] = size(U);

    % -------------------------
    % 0) 基本检查
    % -------------------------
    if size(Y, 1) ~= N
        error('aPC:SizeMismatch', ...
            'U and Y must have the same number of rows.');
    end

    if ~isscalar(ncomp) || ncomp < 1 || ncomp ~= floor(ncomp)
        error('aPC:InvalidNComp', ...
            'ncomp must be a positive integer.');
    end

    % -------------------------
    % 1) U 已标准化，不需要再处理
    % -------------------------
    Uc = U;

    % -------------------------
    % 2) Y 中心化
    % -------------------------
    meanY = mean(Y, 1);
    Yc = Y - meanY;

    % -------------------------
    % 3) Centering matrix H
    % -------------------------
    H = eye(N) - ones(N) / N;

    % -------------------------
    % 4) Label kernel (linear)
    % -------------------------
    Ky = Yc * Yc.';
    Ky = (Ky + Ky.') / 2;

    % -------------------------
    % 5) HSIC core matrix on labels
    % -------------------------
    B = H * Ky * H;
    B = (B + B.') / 2;

    % -------------------------
    % 6) 分解 B = Delta * Delta'
    % -------------------------
    [Qb, Db] = eig(B);
    [bvals, idx_b] = sort(diag(Db), 'descend');
    Qb = Qb(:, idx_b);

    tolB = max(size(B)) * eps(max(abs(bvals)));
    keep_b = bvals > tolB;
    rB = sum(keep_b);

    if rB == 0
        error('aPC:NoEffectiveComponent', ...
            'No positive supervised component found in H*Ky*H.');
    end

    Delta = Qb(:, keep_b) * diag(sqrt(bvals(keep_b)));   % N x rB

    % -------------------------
    % 7) SVD 求解 primal HSIC-SPCA
    % -------------------------
    % M = U' * B * U = (Delta' * U)' * (Delta' * U)
    Psi = Delta.' * Uc;              % rB x d

    [~, S, V] = svd(Psi, 'econ');    % Psi = Upsi * S * V'
    sing_vals = diag(S);
    eigvals = sing_vals .^ 2;        % 对应 M 的特征值

    tolS = max(size(Psi)) * eps(max(sing_vals));
    positive_mask = sing_vals > tolS;
    r_eff = sum(positive_mask);

    if r_eff == 0
        error('aPC:NoEffectiveComponent', ...
            'No positive supervised component found in HSIC-SPCA.');
    end

    r_use = min(ncomp, r_eff);

    % V 的前 r_use 列就是投影方向
    W = V(:, 1:r_use);

    % 数值稳定性：正交化
    [W, ~] = qr(W, 0);

    % -------------------------
    % 8) 降维坐标
    % -------------------------
    X_red = Uc * W;

    % -------------------------
    % 9) 保存模型
    % -------------------------
    ReduceModel = struct();
    ReduceModel.method = 'spca_hsic';

    ReduceModel.params = struct();
    ReduceModel.params.W = W;
    ReduceModel.params.meanY = meanY;
    ReduceModel.params.ncomp_requested = ncomp;
    ReduceModel.params.ncomp_used = r_use;

    ReduceModel.info = struct();
    ReduceModel.info.singular_values = sing_vals;
    ReduceModel.info.eigvals = eigvals;
    ReduceModel.info.effective_rank = r_eff;
    ReduceModel.info.rank_B = rB;

    % 注意：U_new 必须已按训练集方式标准化
    ReduceModel.transform = @(U_new) (U_new * W);

end
