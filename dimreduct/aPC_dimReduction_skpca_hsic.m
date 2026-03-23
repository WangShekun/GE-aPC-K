function [X_red, ReduceModel] = aPC_dimReduction_skpca_hsic(U, Y, ModelOpts)
% aPC_dimReduction_skpca_hsic
% ---------------------------------------------------------
% 按 KernelPCA 的外部适配器风格，实现监督 KPCA (HSIC) 的降维接口。
%
% 内部自动组装散落在 ModelOpts 中的 X-kernel & Y-kernel 字段：
%
%   kernel_type, kernel_gamma, kernel_degree, kernel_offset
%   label_kernel_type, label_kernel_gamma, label_kernel_degree, label_kernel_offset
%
% 输出：
%   X_red       — 训练降维空间 (N × red_dim)
%   ReduceModel — 含 transform，用于 prediction 阶段
% ---------------------------------------------------------

    %% ====== 1) 组装输入核 KernelX（对应 X → Kx） ======

    % 1.1 类型（必须给）
    if ~isfield(ModelOpts,'kernel_type') || isempty(ModelOpts.kernel_type)
        error('SKPCA-HSIC requires ModelOpts.kernel_type such as ''gaussian''.');
    end
    ktype = char(ModelOpts.kernel_type);

    % 1.2 可选参数
    gammaX  = [];
    degreeX = [];
    offsetX = [];

    if isfield(ModelOpts,'kernel_gamma')  && ~isempty(ModelOpts.kernel_gamma)
        gammaX  = ModelOpts.kernel_gamma;
    end
    if isfield(ModelOpts,'kernel_degree') && ~isempty(ModelOpts.kernel_degree)
        degreeX = ModelOpts.kernel_degree;
    end
    if isfield(ModelOpts,'kernel_offset') && ~isempty(ModelOpts.kernel_offset)
        offsetX = ModelOpts.kernel_offset;
    end

    % 1.3 构造输入核 Kernel
    kernelX = Kernel('type',   ktype, ...
                     'gamma',  gammaX, ...
                     'degree', degreeX, ...
                     'offset', offsetX);


    %% ====== 2) 组装监督核 KernelY（对应 Y → Ky） ======

    % 若没有提供，默认用 linear（回归最稳定）
    if ~isfield(ModelOpts,'label_kernel_type') || isempty(ModelOpts.label_kernel_type)
        ltype = 'linear';
    else
        ltype = char(ModelOpts.label_kernel_type);
    end

    gammaY  = [];
    degreeY = [];
    offsetY = [];

    if isfield(ModelOpts,'label_kernel_gamma')  && ~isempty(ModelOpts.label_kernel_gamma)
        gammaY  = ModelOpts.label_kernel_gamma;
    end
    if isfield(ModelOpts,'label_kernel_degree') && ~isempty(ModelOpts.label_kernel_degree)
        degreeY = ModelOpts.label_kernel_degree;
    end
    if isfield(ModelOpts,'label_kernel_offset') && ~isempty(ModelOpts.label_kernel_offset)
        offsetY = ModelOpts.label_kernel_offset;
    end

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
