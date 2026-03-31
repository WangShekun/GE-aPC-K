clear; clc; close all;
% rng(80);

repo_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, 'core'));
aPC

ModelOpts = struct();

% === 基础多项式设置 ===
ModelOpts.Order = [5];                  % 多项式总阶数 p；可设为标量或搜索向量
ModelOpts.qnorm = [0.7];                 % q-norm 截断参数；越小越稀疏
ModelOpts.aPC_Method = 'stieltjes';      % 一维 aPC 构造方法：'2012' | '2018' | 'stieltjes'
ModelOpts.RegressionMethod = 'lars';     % APC trend 的回归器：通常用 'lars'

% === 模型选择设置 ===
ModelOpts.ModelType = 'apck';          % 模型类型：'apc' | 'apck' | 'geapck'
ModelOpts.PCKType = 'optimal';           % PCK 策略：'sequential' | 'optimal'
% ModelOpts.GradientMode = 'provided';     % GEAPCK 梯度来源：'provided' | 'finite-difference' | 'auto'
ModelOpts.Parallel = 0;                  % 是否并行搜索候选配置：0/1

% === 降维设置 ===
ModelOpts.ReductionMethod = 'pls';     % 降维方法：'pls' | 'pca' | 'kpca' | 'mcp' | 'ica' | 'rp' | ...
ModelOpts.red_dim = [6];               % 降维后的目标维数；可设为标量或搜索向量

% === KPCA / SKPCA 输入核设置 ===
% ModelOpts.kpca_opts.kernel_type = {'gaussian'};  % KPCA 输入核类型
% ModelOpts.kpca_opts.kernel_gamma = [0.1];        % KPCA 高斯/拉普拉斯等核尺度参数
% ModelOpts.kpca_opts.kernel_degree = [2];         % KPCA 多项式核次数
% ModelOpts.kpca_opts.kernel_offset = [1];         % KPCA 多项式核偏置项
% ModelOpts.skpca_opts.kernel_type = {'gaussian'}; % SKPCA 输入核类型
% ModelOpts.skpca_opts.kernel_gamma = [0.1];       % SKPCA 输入核尺度参数
% ModelOpts.skpca_opts.kernel_degree = [2];        % SKPCA 输入核次数
% ModelOpts.skpca_opts.kernel_offset = [1];        % SKPCA 输入核偏置项

% === SKPCA 标签核设置 ===
% ModelOpts.skpca_opts.label_kernel_type = {'gaussian'}; % 标签核类型
% ModelOpts.skpca_opts.label_kernel_gamma = [1/128^2];   % 标签核尺度参数
% ModelOpts.skpca_opts.label_kernel_degree = [2];        % 标签核次数
% ModelOpts.skpca_opts.label_kernel_offset = [1];        % 标签核偏置项

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
% ModelOpts.ica_opts.approach = 'symm';       % ICA 求解方式：'symm' | 'defl'
% ModelOpts.ica_opts.fun = 'tanh';            % 非线性函数：'tanh' | 'gauss' | 'pow3'
% ModelOpts.ica_opts.a1 = 1.0;                % tanh 非线性尺度参数
% ModelOpts.ica_opts.epsilon = 1e-4;          % 收敛阈值
% ModelOpts.ica_opts.maxIter = 200;           % 最大迭代次数
% ModelOpts.ica_opts.mu = 1.0;                % 阻尼/步长参数
% ModelOpts.ica_opts.stabilization = 'on';    % 是否启用稳定化
% ModelOpts.ica_opts.whiten_solver = 'svd';   % 白化求解器：'svd' | 'eig'
% ModelOpts.ica_opts.verbose = 0;             % ICA 过程是否输出信息

% === 随机投影设置 ===
% ModelOpts.rp_opts.type = 'gaussian';        % 随机投影矩阵类型：'gaussian' | 'rademacher'
% ModelOpts.rp_opts.scale = 'jl';             % 是否做 Johnson-Lindenstrauss 缩放：'jl' | 'none'
% ModelOpts.rp_opts.seed = 80;                % 随机种子

% === APCK / GEAPCK Kriging 设置 ===
ModelOpts.KrigingOpts.corr = 'gaussian';      % 相关函数类型：'gaussian' | 'exponential' | 'matern32'
ModelOpts.KrigingOpts.theta0 = 1.0;           % 长度尺度初值；标量时自动扩展到各维
ModelOpts.KrigingOpts.theta_lb = 1e-3;        % 长度尺度下界
ModelOpts.KrigingOpts.theta_ub = 1e2;         % 长度尺度上界
ModelOpts.KrigingOpts.nugget = 1e-8;          % nugget 正则项，改善数值稳定性
ModelOpts.KrigingOpts.trend_range = [0.1, 0.8]; % trend 路径比例区间；例如 [0.1,0.8] 表示在前 10%~80% 中选优
ModelOpts.KrigingOpts.optimize_theta = true;  % 是否优化 Kriging 长度尺度
ModelOpts.KrigingOpts.DisplayProgress = true; % 是否显示 sequential/optimal 的内部进度

% === GE 数值梯度回退设置 ===
% ModelOpts.FDOpts.type = 'forward';          % 差分类型：'forward' | 'central'
% ModelOpts.FDOpts.step = 1e-6;               % 相对步长，用于自动数值梯度

uqlab

% InputOpts = ishigami_Input;

InputOpts = wingweight_Input;

% InputOpts = morris10_Input;

% InputOpts = dim100_Input;
% InputOpts = dim150_Input;
% InputOpts = dim200_Input_Case3;
% InputOpts = dim300_Input;

myInput = uq_createInput(InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, 500);
data.Y_train = wingweight(data.X_train);

% Ishigami 解析梯度（用于 GEAPCK）
x1 = data.X_train(:,1);
x2 = data.X_train(:,2);
x3 = data.X_train(:,3);
data.dYdX_train = [ ...
    cos(x1) .* (1 + 0.1 .* x3.^4), ...
    14 .* sin(x2) .* cos(x2), ...
    0.4 .* x3.^3 .* sin(x1) ...
];

data.X_val   = uq_getSample(myInput, 4000);
data.Y_val   = wingweight(data.X_val);

%% 
MetaOpts_OLS.Type = 'MetaModel';
MetaOpts_OLS.MetaType = 'PCE';
MetaOpts_OLS.Method = 'lars';
MetaOpts_OLS.TruncOptions.qNorm = 0.7;
MetaOpts_OLS.Degree = 8;
MetaOpts_OLS.ExpDesign.X = data.X_train;
MetaOpts_OLS.ExpDesign.Y = data.Y_train;
MetaOpts_OLS.Display = 0;
myPCE_OLS = uq_createModel(MetaOpts_OLS);

Y_OLS = uq_evalModel(myPCE_OLS, data.X_val);
RELMSE_OLS = sum((Y_OLS - data.Y_val).^2) / sum((mean(data.Y_val) - data.Y_val).^2)

% PCESobol.Type = 'Sensitivity';
% PCESobol.Method = 'Sobol';
% PCESobol.Sobol.Order = 2;
% PCESobolAnalysis = uq_createAnalysis(PCESobol);
% % uq_display(PCESobolAnalysis);
% uq_print(PCESobolAnalysis);
%% 
tic
[results, ModelOpts, data, ReduceModel] = aPC_Train(data, ModelOpts);
toc
data.Y_pred = aPC_Predict(data.X_val, results, ModelOpts, ReduceModel);

results.RelMSE = sum((data.Y_pred - data.Y_val).^2) / ...
                 sum((data.Y_val - mean(data.Y_val)).^2);

aPC_print(results, ModelOpts, data);
% aPC_plot(data, results, ModelOpts, "all");

%% 不确定性量化（可选）
% UQ = aPC_ComputeUQ(results, data);
% aPC_printUQ(UQ);
% aPC_PlotUQ(UQ, data);

%% 灵敏度分析
% SA = aPC_ComputeSensitivity(results, ModelOpts);
% % aPC_PlotSensitivity(SA, ModelOpts);
% aPC_printSensitivity(SA,ModelOpts);
