function [X_red, ReduceModel] = aPC_dimReduction_rp(U, ModelOpts)
% aPC_dimReduction_rp
% ---------------------------------------------------------
% Random Projection (Johnson–Lindenstrauss) 降维模块
%
% 特点：
%   - 无监督
%   - 线性
%   - 不依赖统计分布
%   - 经典高维基线方法
%
% 输入：
%   U         — 输入样本 (N×M)，建议已标准化
%   ModelOpts — 控制参数
%
% 必需字段：
%   ModelOpts.red_dim     : 目标降维维数 r
%
% 可选字段：
%   ModelOpts.rp_type     : 'gaussian'(默认) | 'rademacher'
%   ModelOpts.rp_scale    : 'jl'(默认) | 'none'
%   ModelOpts.rp_seed     : 随机种子（默认空）
%
% 输出：
%   X_red       — 降维后变量 (N×r)
%   ReduceModel — 含 transform 句柄
% ---------------------------------------------------------

    %% ====== 0) 参数检查 ======
    if ~isfield(ModelOpts,'red_dim') || isempty(ModelOpts.red_dim)
        error('Random Projection requires ModelOpts.red_dim.');
    end
    r = ModelOpts.red_dim;

    [N, M] = size(U);

    %% ====== 1) 读取参数 ======
    rp_type  = local_getopt(ModelOpts, 'rp_type',  'gaussian');
    rp_scale = local_getopt(ModelOpts, 'rp_scale', 'jl');
    rp_seed  = local_getopt(ModelOpts, 'rp_seed',  []);

    if ~isempty(rp_seed)
        rng(rp_seed);
    end

    %% ====== 2) 构造随机投影矩阵 ======
    switch lower(rp_type)
        case 'gaussian'
            % N(0,1)
            R = randn(M, r);

        case 'rademacher'
            % P(R=+1)=P(R=-1)=0.5
            R = sign(rand(M, r) - 0.5);

        otherwise
            error('Unknown rp_type: %s', rp_type);
    end

    %% ====== 3) Johnson–Lindenstrauss 缩放 ======
    switch lower(rp_scale)
        case 'jl'
            % 保距离期望不变
            R = R / sqrt(r);

        case 'none'
            % 不缩放（一般不推荐）
        otherwise
            error('Unknown rp_scale: %s', rp_scale);
    end

    %% ====== 4) 投影 ======
    X_red = U * R;   % (N×r)

    %% ====== 5) ReduceModel 封装 ======
    params = struct();
    params.R = R;

    ReduceModel = struct();
    ReduceModel.method = 'rp';
    ReduceModel.params = params;

    ReduceModel.transform = @(U_new) U_new * params.R;
end

%% ============================================================
% Safe option getter
%% ============================================================
function v = local_getopt(S, field, defaultVal)
    if isfield(S, field) && ~isempty(S.(field))
        v = S.(field);
    else
        v = defaultVal;
    end
end
