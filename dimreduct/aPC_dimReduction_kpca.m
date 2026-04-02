function [X_red, ReduceModel] = aPC_dimReduction_kpca(U, ModelOpts)
% aPC_dimReduction_kpca
% ---------------------------------------------------------
% 按官方 KPCA-MATLAB README 实现的降维适配器。
% 方法参数优先从 ModelOpts.kpca_opts 读取（推荐）：
%
%   ModelOpts.kpca_opts.kernel_type
%   ModelOpts.kpca_opts.kernel_gamma
%   ModelOpts.kpca_opts.kernel_degree
%   ModelOpts.kpca_opts.kernel_offset
%
% 输出：
%   X_red       — 训练降维空间
%   ReduceModel — 含 transform，用于 prediction 阶段
% ---------------------------------------------------------

    %% ====== 1) 组装 Kernel 对象 ======

    opts = struct();
    if isfield(ModelOpts, 'kpca_opts') && ~isempty(ModelOpts.kpca_opts)
        opts = ModelOpts.kpca_opts;
    end

    % 1.1 类型（必须给）
    if ~isfield(opts,'kernel_type') || isempty(opts.kernel_type)
        error('KPCA requires ModelOpts.kpca_opts.kernel_type such as ''gaussian''.');
    end
    ktype = char(opts.kernel_type);

    % 1.2 可选参数
    gamma  = localGetOpt(opts, 'kernel_gamma',  []);
    degree = localGetOpt(opts, 'kernel_degree', []);
    offset = localGetOpt(opts, 'kernel_offset', []);

    % 1.3 构造 Kernel 对象（KPCA-MATLAB 接受 [] 默认值）
    kernel = Kernel('type',   ktype, ...
                    'gamma',  gamma, ...
                    'degree', degree, ...
                    'offset', offset);

    %% ====== 2) 构造 KPCA 参数结构 ======
    rd = ModelOpts.red_dim;
    if rd > 1
        numComponents = rd;      % 指定主成分个数
    elseif rd > 0 && rd < 1
        numComponents = rd;      % 指定解释率阈值
    else
        error('KPCA red_dim must be integer >1 OR a ratio in (0,1).');
    end

    parameter = struct('numComponents', numComponents, ...
                       'kernelFunc',   kernel);

    %% ====== 3) 训练 KPCA（READ ME Example 02） ======
    kpca = KernelPCA(parameter);
    kpca.train(U);

    %% ====== 4) 训练降维结果（READ ME Example 02 的 score 字段） ======
    X_red = kpca.score;

    %% ====== 5) 封装 ReduceModel，供 prediction 用 ======
    ReduceModel = struct();
    ReduceModel.method = 'kpca';
    ReduceModel.kpca   = kpca;

    % prediction：使用 Example 05 (kpca.test)
    ReduceModel.transform = @(U_new) local_kpca_predict(U_new, kpca);
end


%% ============================================================
%         Prediction 阶段：Example 05 的官方投影方式
%% ============================================================
function Z = local_kpca_predict(U_new, kpca)
    result = kpca.test(U_new);
    Z = result.score;
end

function val = localGetOpt(opts, optName, defaultVal)
    if isfield(opts, optName) && ~isempty(opts.(optName))
        val = opts.(optName);
    else
        val = defaultVal;
    end
end
