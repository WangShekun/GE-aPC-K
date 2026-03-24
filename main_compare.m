clear; clc; close all;
% rng(80);

repo_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, 'core'));
aPC
uqlab

ModelOpts = struct();

% === 基础多项式设置 ===
ModelOpts.Order = [8];                  % 多项式总阶数 p；可设为标量或搜索向量
ModelOpts.qnorm = [0.7];                % q-norm 截断参数；越小越稀疏
ModelOpts.aPC_Method = 'stieltjes';     % 一维 aPC 构造方法：'2012' | '2018' | 'stieltjes'
ModelOpts.RegressionMethod = 'lars';    % APC trend 的回归器：通常用 'lars'

% === 模型选择设置 ===
ModelOpts.ModelType = 'apck';           % 模型类型：'apc' | 'apck' | 'geapck'
ModelOpts.PCKType = 'optimal';          % PCK 策略：'sequential' | 'optimal'
% ModelOpts.GradientMode = 'provided';  % GEAPCK 梯度来源：'provided' | 'finite-difference' | 'auto'
ModelOpts.Parallel = 0;                 % 是否并行搜索候选配置：0/1

% === 降维设置 ===
% ModelOpts.ReductionMethod = 'pls';    % 降维方法：'pls' | 'pca' | 'kpca' | 'mcp' | 'ica' | 'rp' | ...
% ModelOpts.red_dim = [4];              % 降维后的目标维数；可设为标量或搜索向量

% === KPCA / SKPCA 输入核设置 ===
% ModelOpts.kernel_type = {'gaussian'}; % 输入核类型
% ModelOpts.kernel_gamma = [0.1];       % 高斯/拉普拉斯等核的尺度参数
% ModelOpts.kernel_degree = [2];        % 多项式核次数
% ModelOpts.kernel_offset = [1];        % 多项式核偏置项

% === SKPCA 标签核设置 ===
% ModelOpts.label_kernel_type = {'gaussian'}; % 标签核类型
% ModelOpts.label_kernel_gamma = [1/128^2];   % 标签核尺度参数

% === MCP 设置 ===
% ModelOpts.mcp_opts.maxIter = 3000;          % MCP 最大迭代次数
% ModelOpts.mcp_opts.whiten = 1;              % 是否对输入先白化
% ModelOpts.mcp_opts.alpha_cov = 1.0;         % 协方差项权重
% ModelOpts.mcp_opts.max_moment_order = [6];  % 矩信息最高阶数
% ModelOpts.mcp_opts.moment_base = 1.0;       % 矩权重基数
% ModelOpts.mcp_opts.moment_decay = [0.6];    % 高阶矩衰减率
% ModelOpts.mcp_opts.lambda_y = [0.05];       % 监督项权重
% ModelOpts.mcp_opts.supervision_mode = 'pls';% 监督模式
% ModelOpts.mcp_opts.manopt_solver = 'rlbfgs';% MCP 优化器

% === ICA 设置 ===
% ModelOpts.ica_approach = 'symm';            % ICA 求解方式：'symm' | 'defl'
% ModelOpts.ica_fun = 'tanh';                 % 非线性函数：'tanh' | 'gauss' | 'pow3'
% ModelOpts.ica_a1 = 1.0;                     % tanh 非线性尺度参数
% ModelOpts.ica_epsilon = 1e-4;               % 收敛阈值
% ModelOpts.ica_maxIter = 200;                % 最大迭代次数
% ModelOpts.ica_mu = 1.0;                     % 阻尼/步长参数
% ModelOpts.ica_stabilization = 'on';         % 是否启用稳定化
% ModelOpts.ica_whiten_solver = 'svd';        % 白化求解器：'svd' | 'eig'
% ModelOpts.ica_verbose = 0;                  % ICA 过程是否输出信息

% === 随机投影设置 ===
% ModelOpts.rp_type = 'gaussian';             % 随机投影矩阵类型：'gaussian' | 'rademacher'
% ModelOpts.rp_scale = 'jl';                  % 是否做 Johnson-Lindenstrauss 缩放：'jl' | 'none'

% === APCK / GEAPCK Kriging 设置 ===
ModelOpts.KrigingOpts.corr = 'gaussian';      % 相关函数类型：'gaussian' | 'exponential' | 'matern32'
ModelOpts.KrigingOpts.theta0 = 1.0;           % 长度尺度初值；标量时自动扩展到各维
ModelOpts.KrigingOpts.theta_lb = 1e-3;        % 长度尺度下界
ModelOpts.KrigingOpts.theta_ub = 1e2;         % 长度尺度上界
ModelOpts.KrigingOpts.nugget = 1e-8;          % nugget 正则项，改善数值稳定性
ModelOpts.KrigingOpts.max_trend_terms = [0.1, 0.8]; % trend 路径比例区间
ModelOpts.KrigingOpts.optimize_theta = true;  % 是否优化 Kriging 长度尺度
ModelOpts.KrigingOpts.DisplayProgress = true;% 是否显示 sequential/optimal 的内部进度

% === GE 数值梯度回退设置 ===
% ModelOpts.FDOpts.type = 'forward';          % 差分类型：'forward' | 'central'
% ModelOpts.FDOpts.step = 1e-6;               % 相对步长，用于自动数值梯度

% === 基准函数选择（含梯度） ===
Benchmark.Name = 'ishigami';          % 'ishigami' | 'wingweight' | 'morris10'
Benchmark.N_train = 300;
Benchmark.N_val = 2000;
Benchmark.GradientSource = 'finite-difference';  % 'analytic' | 'finite-difference' | 'auto'
Benchmark.FDType = 'central';           % 'forward' | 'central'
Benchmark.FDStep = 1e-6;                % 差分相对步长

benchSpec = get_benchmark_spec(Benchmark.Name);

myInput = uq_createInput(benchSpec.InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, Benchmark.N_train);
data.Y_train = benchSpec.ModelFcn(data.X_train);
data.ModelFcn = benchSpec.ModelFcn;

switch lower(strtrim(Benchmark.GradientSource))
    case 'analytic'
        if isempty(benchSpec.GradFcn)
            error('Benchmark %s does not provide analytic gradient.', Benchmark.Name);
        end
        data.dYdX_train = benchSpec.GradFcn(data.X_train);
    case 'finite-difference'
        % Leave dYdX_train empty on purpose; GEAPCK will call aPC_FDGradients.
    case 'auto'
        if ~isempty(benchSpec.GradFcn)
            data.dYdX_train = benchSpec.GradFcn(data.X_train);
        end
    otherwise
        error('Unsupported GradientSource: %s', Benchmark.GradientSource);
end

data.X_val = uq_getSample(myInput, Benchmark.N_val);
data.Y_val = benchSpec.ModelFcn(data.X_val);

model_list = {'uqlab_pce', 'apc', 'apck', 'geapck'};
rows = numel(model_list);

bench = repmat(struct( ...
    'Name', '', ...
    'TrainSec', NaN, ...
    'PredictSec', NaN, ...
    'TotalSec', NaN, ...
    'RelMSE', NaN, ...
    'LOO', NaN, ...
    'Status', ''), rows, 1);

fprintf('\n==================== Compare Start ====================\n');
fprintf('Benchmark: %s | GradientSource: %s\n', upper(Benchmark.Name), upper(Benchmark.GradientSource));
for i = 1:rows
    name = model_list{i};
    fprintf('\n[%d/%d] %s\n', i, rows, upper(name));
    try
        switch name
            case 'uqlab_pce'
                out = run_uqlab_pce(data, ModelOpts);
            otherwise
                out = run_apc_family(data, ModelOpts, name, Benchmark);
        end
        bench(i) = out;
        bench(i).Status = 'OK';
        fprintf('  RelMSE = %.6e | Train = %.3fs | Pred = %.3fs\n', ...
            out.RelMSE, out.TrainSec, out.PredictSec);
    catch ME
        bench(i).Name = upper(name);
        bench(i).Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

fprintf('\n==================== Compare Summary ====================\n');
fprintf('%-10s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
    'Model', 'Train(s)', 'Pred(s)', 'Total(s)', 'RelMSE', 'LOO', 'Status');
fprintf('%s\n', repmat('-', 1, 96));
for i = 1:rows
    fprintf('%-10s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
        bench(i).Name, fmt_num(bench(i).TrainSec), fmt_num(bench(i).PredictSec), ...
        fmt_num(bench(i).TotalSec), fmt_num(bench(i).RelMSE), fmt_num(bench(i).LOO), ...
        bench(i).Status);
end
fprintf('%s\n', repmat('=', 1, 62));


function out = run_uqlab_pce(data, ModelOpts)
    MetaOpts = struct();
    MetaOpts.Type = 'MetaModel';
    MetaOpts.MetaType = 'PCE';
    MetaOpts.Method = lower(ModelOpts.RegressionMethod);
    MetaOpts.TruncOptions.qNorm = ModelOpts.qnorm(1);
    MetaOpts.Degree = ModelOpts.Order(1);
    MetaOpts.ExpDesign.X = data.X_train;
    MetaOpts.ExpDesign.Y = data.Y_train;
    MetaOpts.Display = 0;

    t_train = tic;
    model = uq_createModel(MetaOpts);
    train_sec = toc(t_train);

    t_pred = tic;
    y_pred = uq_evalModel(model, data.X_val);
    pred_sec = toc(t_pred);

    out = pack_result('UQLAB-PCE', train_sec, pred_sec, calc_relmse(y_pred, data.Y_val), NaN);
end


function out = run_apc_family(data, ModelOpts, model_name, Benchmark)
    opts = ModelOpts;
    opts.ModelType = lower(model_name);

    if strcmpi(opts.ModelType, 'apc')
        if isfield(opts, 'PCKType')
            opts = rmfield(opts, 'PCKType');
        end
    elseif strcmpi(opts.ModelType, 'geapck')
        src = lower(strtrim(Benchmark.GradientSource));
        if strcmp(src, 'analytic')
            opts.GradientMode = 'provided';
        elseif strcmp(src, 'finite-difference')
            opts.GradientMode = 'finite-difference';
        else
            opts.GradientMode = 'auto';
        end

        if ~isfield(opts, 'FDOpts') || isempty(opts.FDOpts)
            opts.FDOpts = struct();
        end
        opts.FDOpts.type = Benchmark.FDType;
        opts.FDOpts.step = Benchmark.FDStep;
    end

    t_train = tic;
    [results, opts_out, ~, ReduceModel] = aPC_Train(data, opts);
    train_sec = toc(t_train);

    t_pred = tic;
    y_pred = aPC_Predict(data.X_val, results, opts_out, ReduceModel);
    pred_sec = toc(t_pred);

    loo = NaN;
    if isfield(results, 'LOO')
        loo = results.LOO;
    elseif isfield(results, 'optErrorParams') && isfield(results.optErrorParams, 'loo')
        loo = results.optErrorParams.loo;
    end

    out = pack_result(upper(model_name), train_sec, pred_sec, calc_relmse(y_pred, data.Y_val), loo);
end


function out = pack_result(name, train_sec, pred_sec, relmse, loo)
    out = struct();
    out.Name = name;
    out.TrainSec = train_sec;
    out.PredictSec = pred_sec;
    out.TotalSec = train_sec + pred_sec;
    out.RelMSE = relmse;
    out.LOO = loo;
    out.Status = '';
end


function e = calc_relmse(y_pred, y_true)
    y_pred = y_pred(:);
    y_true = y_true(:);
    e = sum((y_pred - y_true).^2) / sum((y_true - mean(y_true)).^2);
end


function s = fmt_num(x)
    if isempty(x) || ~isfinite(x)
        s = '-';
        return;
    end
    if abs(x) >= 1e3 || abs(x) < 1e-2
        s = sprintf('%.3e', x);
    else
        s = sprintf('%.4f', x);
    end
end


function spec = get_benchmark_spec(name)
    key = lower(strtrim(name));
    switch key
        case 'ishigami'
            spec.InputOpts = ishigami_Input;
            spec.ModelFcn = @ishigami;
            spec.GradFcn = @ishigami_grad;

        case 'wingweight'
            spec.InputOpts = wingweight_Input;
            spec.ModelFcn = @wingweight;
            spec.GradFcn = @wingweight_grad;

        case 'morris10'
            spec.InputOpts = morris10_Input;
            spec.ModelFcn = @morris10;
            spec.GradFcn = @morris10_grad;

        otherwise
            error('Unsupported benchmark: %s', name);
    end
end
