function [data, ModelOpts, ReduceModel] = aPC_dimReduction(data, ModelOpts)
% aPC_dimReduction
% --------------------------------------------------------------
% 通用降维总控：
%   - 无 ReductionMethod  : 不降维，原样返回（ReduceModel = []）
%   - 'pls'               : 调 aPC_dimReduction_pls()
%   - 'pca'               : 调 aPC_dimReduction_pca()
%   - 'kpca'              : 调 aPC_dimReduction_kpca()
%
% 统一做法：
%   1) 在子函数内完成“方法本身的降维映射”，得到 X_red (N×r)
%   2) 在本函数内对 X_red 做“二次标准化”（0 均值、1 标准差）
%   3) 将二次标准化参数 (mu2, sigma2) 封装进 ReduceModel.transform
%      => 预测阶段：先子方法 transform，再做同样的标准化
%
% 依赖（建议在 dimreduct/ 目录下）：
%   aPC_dimReduction_pls.m
%   aPC_dimReduction_pca.m
%   aPC_dimReduction_kpca.m            % 适配外部 KPCA 库
%
% 输入：
%   data.U        - 已在上游标准化的输入 (N×M)
%   data.Y_train  - 输出 (N×1)（PLS 需要）
%   ModelOpts.ReductionMethod  - 'pls' | 'pca' | 'kpca' | []/缺省
%   ModelOpts.red_dim          - 降维后维数
%   方法专属参数统一在各方法内部读取（例如 mcp_opts/kpca_opts/ica_opts/rp_opts）
%
% 输出：
%   data.U        - 若降维，则为“降维+二次标准化”后的特征 (N×r)
%   ModelOpts.num_vars - 同步更新为 r
%   ReduceModel   - 结构体：
%       .method
%       .transform(U_new)   % 先做方法映射，再做二次标准化
%       .post_standardize   % true/false
%       .mu2, .sigma2       % 二次标准化参数
%       .params/.info/...   % 子方法返回的其它信息
% --------------------------------------------------------------

    % ====== 0) 无降维：直接返回 ======
    if ~isfield(ModelOpts, 'ReductionMethod') || isempty(ModelOpts.ReductionMethod)
        ReduceModel = [];
        % 不改 data.U 与 ModelOpts.num_vars
        return;
    end

    if ~isfield(data, 'U') || isempty(data.U)
        error('aPC_dimReduction: data.U 为空。请在此之前完成外层标准化（map_to_U）。');
    end

    U = data.U;
    Y = [];
    if isfield(data, 'Y_train'), Y = data.Y_train; end

    method = lower(ModelOpts.ReductionMethod);

    % ====== 1) 分发到具体方法，获得 X_red (N×r) 与 ReduceModel（含 transform）======
    switch method
        case 'pls'
            if isempty(Y), error('PLS 降维需要 data.Y_train。'); end
            [X_red, ReduceModel] = aPC_dimReduction_pls(U, Y, ModelOpts.red_dim);

        case 'pca'
            [X_red, ReduceModel] = aPC_dimReduction_pca(U, ModelOpts.red_dim);

        case 'kpca'
            % 适配外部 KPCA 库（Kernel/KernelPCA）
            [X_red, ReduceModel] = aPC_dimReduction_kpca(U, ModelOpts);

        case 'spca_hsic'
            [X_red, ReduceModel] = aPC_dimReduction_spca_hsic(U, Y, ModelOpts.red_dim);

        case 'skpca_hsic'
            % KPCA修改得到的SPKCA
            [X_red, ReduceModel] = aPC_dimReduction_skpca_hsic(U, Y, ModelOpts);

        case 'mcp'
            [X_red, ReduceModel] = aPC_dimReduction_mcp(U, Y, ModelOpts);

        case 'ica'
            [X_red, ReduceModel] = aPC_dimReduction_ica(U, ModelOpts);

        case 'rp'
            [X_red, ReduceModel] = aPC_dimReduction_rp(U, ModelOpts);

        case 'as_pls'
            if isempty(Y), error('AS-PLS 降维需要 data.Y_train。'); end
            if ~isfield(data, 'dYdU_train') || isempty(data.dYdU_train)
                error('aPC:ReductionGradientRequired', 'as_pls requires precomputed gradients.');
            end
            [X_red, ReduceModel] = aPC_dimReduction_as_pls(U, Y, data.dYdU_train, ModelOpts.red_dim, ModelOpts);

        case 'as'
            if ~isfield(data, 'dYdU_train') || isempty(data.dYdU_train)
                error('aPC:ReductionGradientRequired', 'as requires precomputed gradients.');
            end
            [X_red, ReduceModel] = aPC_dimReduction_as(U, data.dYdU_train, ModelOpts.red_dim);

        case 'sas'
            if isempty(Y), error('SAS 降维需要 data.Y_train。'); end
            if ~isfield(data, 'dYdU_train') || isempty(data.dYdU_train)
                error('aPC:ReductionGradientRequired', 'sas requires precomputed gradients.');
            end
            [X_red, ReduceModel] = aPC_dimReduction_sas(U, Y, data.dYdU_train, ModelOpts.red_dim, ModelOpts);

        otherwise
            error('Unknown ReductionMethod: %s', ModelOpts.ReductionMethod);
    end

    % 容错：子函数必须返回 transform 句柄（预测用）
    if ~isfield(ReduceModel, 'transform') || isempty(ReduceModel.transform)
        error('ReduceModel.transform 丢失：请检查子降维函数的返回。');
    end

    % ====== 2) 统一的“降维后二次标准化” ======
    mu2    = mean(X_red, 1);
    sigma2 = std(X_red, 0, 1);
    sigma2(sigma2 == 0) = 1;  % 防零

    X_red_std = (X_red - mu2) ./ sigma2;

    % 覆盖训练特征，并同步维度
    data.U = X_red_std;
    ModelOpts.num_vars = size(X_red_std, 2);

    % ====== 3) 将二次标准化封装进 transform（预测时自动复用） ======
    prev_tf = ReduceModel.transform;  % 原始方法的映射：U_new → Z
    ReduceModel.transform = @(U_new) post_std(prev_tf(U_new), mu2, sigma2);

    ReduceModel.post_standardize = true;
    ReduceModel.mu2   = mu2;
    ReduceModel.sigma2= sigma2;

    % （可选）记录链路信息
    % if ~isfield(ReduceModel, 'info') || isempty(ReduceModel.info)
    %     ReduceModel.info = struct();
    % end
    % ReduceModel.info.post_std_mu    = mu2;
    % ReduceModel.info.post_std_sigma = sigma2;
end


% ====== 局部函数：后标准化 ======
function Zs = post_std(Z, mu, sigma)
    if any(sigma==0), sigma(sigma==0)=1; end
    Zs = (Z - mu) ./ sigma;
end
