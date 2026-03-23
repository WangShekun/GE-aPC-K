clear; clc; close all;
rng(80);

aPC

ModelOpts = struct();
ModelOpts.Order    = [14];      % 最大多项式阶数
ModelOpts.qnorm    = [0.9];       % q-norm 

% ModelOpts.ReductionMethod = 'mcp';
% ModelOpts.red_dim  = [5];

% ModelOpts.kernel_type   = {'gaussian'};
% ModelOpts.kernel_gamma  = [0.1];
% ModelOpts.kernel_degree = [2];
% ModelOpts.kernel_offset = [1];

% ModelOpts.label_kernel_type = {'gaussian'};
% ModelOpts.label_kernel_gamma  = [1/128^2];

% ModelOpts.mcp_opts.maxIter = 3000;
% ModelOpts.mcp_opts.whiten = 1;
% ModelOpts.mcp_opts.alpha_cov = 1.0;
% ModelOpts.mcp_opts.max_moment_order = [6];
% ModelOpts.mcp_opts.moment_base = 1.0;
% ModelOpts.mcp_opts.moment_decay = [0.6];
% ModelOpts.mcp_opts.lambda_y = [0.05];
% ModelOpts.mcp_opts.supervision_mode = 'pls';
% ModelOpts.mcp_opts.manopt_solver = 'rlbfgs';

% ModelOpts.ica_approach      = 'symm';
% ModelOpts.ica_fun           = 'tanh';
% ModelOpts.ica_a1            = 1.0;        % tanh 尺度
% ModelOpts.ica_epsilon       = 1e-4;
% ModelOpts.ica_maxIter       = 200;
% ModelOpts.ica_mu             = 1.0;
% ModelOpts.ica_stabilization  = 'on';
% ModelOpts.ica_whiten_solver = 'svd';
% ModelOpts.ica_verbose       = 0;

% ModelOpts.rp_type         = 'gaussian';   % 默认 & 最常见
% ModelOpts.rp_scale        = 'jl';          % Johnson–Lindenstrauss 缩放

ModelOpts.aPC_Method = 'stieltjes';      % 可选 '2012' | '2018' | 'stieltjes'
ModelOpts.RegressionMethod = 'lars';
ModelOpts.Parallel = 0;

uqlab

InputOpts = ishigami_Input;

% InputOpts = wingweight_Input;

% InputOpts = morris10_Input;

% InputOpts = dim100_Input;
% InputOpts = dim150_Input;
% InputOpts = dim200_Input_Case3;
% InputOpts = dim300_Input;

myInput = uq_createInput(InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, 2000);
data.Y_train = ishigami(data.X_train);

data.X_val   = uq_getSample(myInput, 10000);
data.Y_val   = ishigami(data.X_val);

%% 
MetaOpts_OLS.Type = 'MetaModel';
MetaOpts_OLS.MetaType = 'PCE';
MetaOpts_OLS.Method = 'lars';
MetaOpts_OLS.TruncOptions.qNorm = 0.9;
MetaOpts_OLS.Degree = 14;
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

%% 不确定性量化
UQ = aPC_ComputeUQ(results, data);
aPC_printUQ(UQ);
% aPC_PlotUQ(UQ, data);

aPC_error = [abs(UQ.mean-mean(data.Y_val))/mean(data.Y_val) ...
    ;abs(UQ.var-var(data.Y_val))/var(data.Y_val)]
PCE_error = [abs(myPCE_OLS.PCE.Moments.Mean-mean(data.Y_val))/mean(data.Y_val) ...
    ;abs(myPCE_OLS.PCE.Moments.Var-var(data.Y_val))/var(data.Y_val)]

%% 灵敏度分析
% SA = aPC_ComputeSensitivity(results, ModelOpts);
% % aPC_PlotSensitivity(SA, ModelOpts);
% aPC_printSensitivity(SA,ModelOpts);
