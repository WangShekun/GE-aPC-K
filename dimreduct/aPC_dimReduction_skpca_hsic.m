function [X_red, ReduceModel] = aPC_dimReduction_skpca_hsic(U, Y, ModelOpts)
% aPC_dimReduction_skpca_hsic
% ---------------------------------------------------------
% 按 KernelPCA 的外部适配器风格，实现监督 KPCA (HSIC) 的降维接口。
%
% 方法参数优先从 ModelOpts.skpca_opts 读取（推荐）：
%
%   kernel_type, kernel_gamma, kernel_degree, kernel_offset
%   label_kernel_type, label_kernel_gamma, label_kernel_degree, label_kernel_offset
%
% 输出：
%   X_red       — 训练降维空间 (N × red_dim)
%   ReduceModel — 含 transform，用于 prediction 阶段
% ---------------------------------------------------------

    opts = struct();
    if isfield(ModelOpts, 'skpca_opts') && ~isempty(ModelOpts.skpca_opts)
        opts = ModelOpts.skpca_opts;
    end

    %% ====== 1) 组装输入核 KernelX（对应 X → Kx） ======

    % 1.1 类型（必须给）
    if ~isfield(opts,'kernel_type') || isempty(opts.kernel_type)
        error('SKPCA-HSIC requires ModelOpts.skpca_opts.kernel_type such as ''gaussian''.');
    end
    ktype = char(opts.kernel_type);

    % 1.2 可选参数
    gammaX  = localGetOpt(opts, 'kernel_gamma',  []);
    degreeX = localGetOpt(opts, 'kernel_degree', []);
    offsetX = localGetOpt(opts, 'kernel_offset', []);

    % 1.3 构造输入核 Kernel
    kernelX = Kernel('type',   ktype, ...
                     'gamma',  gammaX, ...
                     'degree', degreeX, ...
                     'offset', offsetX);


    %% ====== 2) 组装监督核 KernelY（对应 Y → Ky） ======

    % 若没有提供，默认用 linear（回归最稳定）
    if ~isfield(opts,'label_kernel_type') || isempty(opts.label_kernel_type)
        ltype = 'linear';
    else
        ltype = char(opts.label_kernel_type);
    end

    gammaY  = localGetOpt(opts, 'label_kernel_gamma',  []);
    degreeY = localGetOpt(opts, 'label_kernel_degree', []);
    offsetY = localGetOpt(opts, 'label_kernel_offset', []);

    kernelY = Kernel('type',   ltype, ...
                     'gamma',  gammaY, ...
                     'degree', degreeY, ...
                     'offset', offsetY);


    %% ====== 3) 解析降维维数 ======
    rd = ModelOpts.red_dim;
    if rd > 1
        numComponents = rd;         % 指定主成分数
    elseif rd > 0 && rd < 1
        numComponents = rd;         % 指定解释率
    else
        error('SKPCA-HSIC red_dim must be integer >1 OR a ratio in (0,1).');
    end

    %% ====== 4) 构造 SKPCA 参数 ======
    parameter = struct('numComponents', numComponents, ...
                       'kernelFunc',   kernelX, ...
                       'kernelY',      kernelY);

    %% ====== 5) 训练 SupervisedKPCA ======
    skpca = SupervisedKernelPCA(parameter);
    skpca.train(U, Y);

    %% ====== 6) 降维结果（score） ======
    X_red = skpca.score;

    %% ====== 7) 封装 ReduceModel，供 prediction 用 ======
    ReduceModel = struct();
    ReduceModel.method = 'skpca_hsic';
    ReduceModel.skpca  = skpca;

    ReduceModel.transform = @(U_new) local_skpca_predict(U_new, skpca);
end


%% ============================================================
%         Prediction 阶段：与 KPCA 完全同风格
%% ============================================================
function Z = local_skpca_predict(U_new, skpca)
    result = skpca.test(U_new);
    Z = result.score;
end

function val = localGetOpt(opts, optName, defaultVal)
    if isfield(opts, optName) && ~isempty(opts.(optName))
        val = opts.(optName);
    else
        val = defaultVal;
    end
end
