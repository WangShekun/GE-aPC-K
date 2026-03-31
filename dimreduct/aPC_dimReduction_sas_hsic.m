function [X_red, ReduceModel] = aPC_dimReduction_sas_hsic(U, G, Y, ncomp, ModelOpts)
% ===============================================================
%  aPC_dimReduction_sas
%
%  加性融合型监督式活跃子空间 (Supervised Active Subspace)
%
%  输入：
%     U         — N×d 输入样本矩阵（建议已做 Z-score 标准化）
%     G         — N×d 梯度矩阵（不要标准化，保留原始物理偏导）
%     Y         — N×1 目标响应
%     ncomp     — 目标降维维度
%     ModelOpts — 可选参数结构体；可通过 ModelOpts.sas_opts 传递：
%                   .alpha  : AS 项权重，默认 1.0
%                   .beta   : 监督项权重，默认 0.2
%                   .normalize_terms : 是否做迹归一化，默认 true
%
%  输出：
%     X_red        — 降维后的样本坐标（N×r_use）
%     ReduceModel  — 模型结构（含 W 和 transform）
%
%  方法：
%     C_as  = (1/N) * G' * G
%     c_xy  = U' * y_c
%     C_y   = c_xy * c_xy'
%     C_sup = alpha * C_as + beta * C_y
%
%  若 normalize_terms = true，则先做：
%     C_as <- C_as / trace(C_as)
%     C_y  <- C_y  / trace(C_y)
%
% ===============================================================

    [N, d] = size(U);
    [Ng, dg] = size(G);

    % -------------------------
    % 0) 默认参数
    % -------------------------
    if nargin < 5 || isempty(ModelOpts)
        ModelOpts = struct();
    end

    opts = struct();
    if isfield(ModelOpts, 'sas_opts') && ~isempty(ModelOpts.sas_opts)
        opts = ModelOpts.sas_opts;
    end

    alpha = localGetOption(opts, 'alpha', 1.0);
    beta = localGetOption(opts, 'beta', 0.2);
    normalize_terms = localGetOption(opts, 'normalize_terms', true);

    % -------------------------
    % 1) 基本检查
    % -------------------------
    if N ~= Ng || d ~= dg
        error('aPC:DimensionMismatch', ...
            'U and G must have exactly the same size (N x d).');
    end

    if size(Y,1) ~= N
        error('aPC:SizeMismatch', ...
            'Y must have the same number of rows as U.');
    end

    if size(Y,2) ~= 1
        error('aPC:InvalidY', ...
            'This version of supervised AS currently assumes scalar output Y (N x 1).');
    end

    if ~isscalar(ncomp) || ncomp < 1 || ncomp ~= floor(ncomp)
        error('aPC:InvalidNComp', ...
            'ncomp must be a positive integer.');
    end

    % -------------------------
    % 2) Y 中心化
    % -------------------------
    meanY = mean(Y, 1);
    yc = Y - meanY;

    % -------------------------
    % 3) 经典 AS 项
    % -------------------------
    C_as = (G.' * G) / N;
    C_as = (C_as + C_as.') / 2;

    % -------------------------
    % 4) 监督项
    % -------------------------
    % c_xy 相当于输入与输出的相关方向
    c_xy = U.' * yc;          % d x 1
    C_y = c_xy * c_xy.';      % d x d
    C_y = (C_y + C_y.') / 2;

    % -------------------------
    % 5) 可选归一化
    % -------------------------
    if normalize_terms
        tr_as = trace(C_as);
        tr_y  = trace(C_y);

        if tr_as > 0
            C_as = C_as / tr_as;
        end
        if tr_y > 0
            C_y = C_y / tr_y;
        end
    end

    % -------------------------
    % 6) 融合矩阵
    % -------------------------
    C_sup = alpha * C_as + beta * C_y;
    C_sup = (C_sup + C_sup.') / 2;

    % -------------------------
    % 7) 特征分解
    % -------------------------
    [V_raw, D_raw] = eig(C_sup);
    [eigvals, idx] = sort(diag(D_raw), 'descend');
    V_raw = V_raw(:, idx);

    tol = max(size(C_sup)) * eps(max(abs(eigvals)));
    positive_mask = eigvals > tol;
    r_eff = sum(positive_mask);

    if r_eff == 0
        error('aPC:NoEffectiveComponent', ...
            'No effective supervised active direction found.');
    end

    r_use = min(ncomp, r_eff);

    W = V_raw(:, 1:r_use);

    % 数值稳定性
    [W, ~] = qr(W, 0);

    % -------------------------
    % 8) 输入样本投影
    % -------------------------
    X_red = U * W;

    % -------------------------
    % 9) 保存模型
    % -------------------------
    ReduceModel = struct();
    ReduceModel.method = 'sas';

    ReduceModel.params = struct();
    ReduceModel.params.W = W;
    ReduceModel.params.alpha = alpha;
    ReduceModel.params.beta = beta;
    ReduceModel.params.normalize_terms = normalize_terms;
    ReduceModel.params.meanY = meanY;
    ReduceModel.params.ncomp_requested = ncomp;
    ReduceModel.params.ncomp_used = r_use;

    ReduceModel.info = struct();
    ReduceModel.info.eigvals = eigvals;
    ReduceModel.info.effective_rank = r_eff;
    ReduceModel.info.C_as = C_as;
    ReduceModel.info.C_y = C_y;
    ReduceModel.info.C_sup = C_sup;

    % 注意：U_new 应使用与训练集一致的标准化方式
    ReduceModel.transform = @(U_new) (U_new * W);

end

function val = localGetOption(opts, name, defaultVal)
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = defaultVal;
    end
end

% function [X_red, ReduceModel] = aPC_dimReduction_sas_hsic(U, G, Y, ncomp)
% % ===============================================================
% %  aPC_dimReduction_sas_hsic
% %
% %  基于 HSIC 准则的监督式活跃子空间 (Supervised Active Subspaces)
% %  利用 SVD 统一求解引擎，自动适应极高维度的基函数/输入。
% %
% %  输入：
% %     U     — N×d 输入样本矩阵（必须已做 Z-score 标准化）
% %     G     — N×d 梯度矩阵（切勿标准化，保留原始物理偏导数值）
% %     Y     — N×1 或 N×k 目标响应/标签
% %     ncomp — 期望的目标降维维度
% %
% %  输出：
% %     X_red        — 降维后的输入样本数据（N×r_use）
% %     ReduceModel  — 模型结构（包含投影矩阵 W 及 transform 回调）
% %
% %  数学原理：
% %     寻找投影矩阵 W，使得梯度投影后的 HSIC 依赖度最大化。
% %     利用 RBF 核打破标量 Y 的秩坍缩限制，允许提取多维活跃子空间。
% %     Psi = G^T * Delta，对 Psi 进行经济型 SVD 即可得到活跃方向 W。
% % ===============================================================
% 
%     [N, d] = size(U);
%     [Ng, dg] = size(G);
% 
%     % -------------------------
%     % 0) 严格的维度一致性检查
%     % -------------------------
%     if N ~= Ng || d ~= dg
%         error('aPC:DimensionMismatch', '样本矩阵 U 和梯度矩阵 G 的维度必须完全一致 (N x d)。');
%     end
%     if size(Y, 1) ~= N
%         error('aPC:SizeMismatch', 'Y 的行数必须与 U 的样本数 N 相同。');
%     end
%     if ~isscalar(ncomp) || ncomp < 1 || ncomp ~= floor(ncomp)
%         error('aPC:InvalidNComp', 'ncomp 必须是正整数。');
%     end
% 
%     % -------------------------
%     % 1) Y 的中心化处理
%     % -------------------------
%     meanY = mean(Y, 1);
%     Yc = Y - meanY;
% 
%     % -------------------------
%     % 2) 构造监督 RBF 核矩阵 (打破 1维 限制的核心所在)
%     % -------------------------
%     % 计算样本间目标值的欧氏距离平方
%     D_sq = pdist2(Yc, Yc, 'squaredeuclidean');
% 
%     % 启发式选取高斯核的带宽参数 (gamma)
%     median_dist = median(D_sq(D_sq > 0)); 
%     if isempty(median_dist) || median_dist == 0 || isnan(median_dist)
%         gamma = 1.0; % 退化情况的兜底保护
%     else
%         gamma = 1 / (2 * median_dist);
%     end
% 
%     % 生成满秩的非线性核矩阵 Ky
%     Ky = exp(-gamma * D_sq);
%     Ky = (Ky + Ky.') / 2; % 保证绝对的数值对称
% 
%     % -------------------------
%     % 3) 提取监督信息的本征结构 (Delta)
%     % -------------------------
%     % 构造中心化矩阵 H
%     H = eye(N) - ones(N) / N;
% 
%     % 计算监督信息主矩阵 B = H * Ky * H
%     B = H * Ky * H;
%     B = (B + B.') / 2;
% 
%     % 对 N x N 的矩阵 B 进行特征分解
%     [Qb, Db] = eig(B);
%     [bvals, idx_b] = sort(diag(Db), 'descend');
%     Qb = Qb(:, idx_b);
% 
%     % 剔除数值噪声特征值
%     tolB = max(size(B)) * eps(max(abs(bvals)));
%     keep_b = bvals > tolB;
%     rB = sum(keep_b);
% 
%     if rB == 0
%         error('aPC:NoEffectiveComponent', '在监督信息矩阵 H*Ky*H 中未找到有效的正特征分量。');
%     end
% 
%     % 提取本征结构 Delta (N x rB)
%     Delta = Qb(:, keep_b) * diag(sqrt(bvals(keep_b)));   
% 
%     % -------------------------
%     % 4) SVD 引擎求解监督式活跃方向 (彻底告别 d x d 庞大矩阵)
%     % -------------------------
%     % 注意：这里用的是梯度矩阵 G，而不是样本 U
%     Psi = G.' * Delta;               % 维度: d x rB
% 
%     % 经济型 SVD 自动选择最优计算路径
%     [U_svd, S, ~] = svd(Psi, 'econ');  
%     sing_vals = diag(S);
%     eigvals_sas = sing_vals .^ 2;    % 这就是传统的“活跃特征值”
% 
%     % -------------------------
%     % 5) 有效维数检查与截断
%     % -------------------------
%     tolS = max(size(Psi)) * eps(max(sing_vals));
%     positive_mask = sing_vals > tolS;
%     r_eff = sum(positive_mask);
% 
%     if r_eff == 0
%         error('aPC:NoEffectiveComponent', '无法从梯度矩阵中提取出有效的监督式活跃方向。');
%     end
% 
%     r_use = min(ncomp, r_eff);
% 
%     % SVD 的左奇异向量即为所求的投影方向 W
%     W = U_svd(:, 1:r_use);
% 
%     % 为确保大规模基函数运算的正交稳定性，进行 QR 兜底
%     [W, ~] = qr(W, 0);
% 
%     % -------------------------
%     % 6) 物理降维 (将投影矩阵作用于原始输入样本 U)
%     % -------------------------
%     X_red = U * W;
% 
%     % -------------------------
%     % 7) 打包并保存模型参数
%     % -------------------------
%     ReduceModel = struct();
%     ReduceModel.method = 'sas';
% 
%     ReduceModel.params = struct();
%     ReduceModel.params.W = W;
%     ReduceModel.params.gamma = gamma; % 记录选用的 RBF 参数
%     ReduceModel.params.meanY = meanY;
%     ReduceModel.params.ncomp_requested = ncomp;
%     ReduceModel.params.ncomp_used = r_use;
% 
%     ReduceModel.info = struct();
%     ReduceModel.info.sas_eigenvalues = eigvals_sas;
%     ReduceModel.info.effective_rank_G = r_eff;
%     ReduceModel.info.rank_B = rB;
% 
%     % 提供给新样本测试的匿名映射函数
%     % 注意：传入的 U_new 必须与训练集 U 使用相同的 Z-score 均值和方差进行过标准化
%     ReduceModel.transform = @(U_new) (U_new * W);
% 
% end
