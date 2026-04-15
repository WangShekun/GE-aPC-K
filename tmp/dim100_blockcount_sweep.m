clear; clc;
rng(80);

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'core'));
aPC;
uqlab;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 0.8;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'geapck_block';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'finite-difference';
ModelOpts.Parallel = 0;
ModelOpts.ReductionMethod = 'pls';
ModelOpts.red_dim = 15;

ModelOpts.KrigingOpts.corr = 'gaussian';
ModelOpts.KrigingOpts.theta0 = 1.0;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 1e2;
ModelOpts.KrigingOpts.nugget = 1e-8;
ModelOpts.KrigingOpts.trend_range = [0.8];
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.DisplayProgress = true;

ModelOpts.FDOpts.type = 'central';
ModelOpts.FDOpts.step = 1e-6;

Benchmark = struct();
Benchmark.N_train = 200;
Benchmark.N_val = 2000;

myInput = uq_createInput(dim100_Input);
data = struct();
data.X_train = uq_getSample(myInput, Benchmark.N_train);
data.Y_train = dim100(data.X_train);
data.ModelFcn = @dim100;
data.X_val = uq_getSample(myInput, Benchmark.N_val);
data.Y_val = dim100(data.X_val);

block_list = [2, 3, 5];
bench = repmat(struct('NumBlocks', NaN, 'TrainSec', NaN, 'PredictSec', NaN, ...
    'TotalSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', ''), numel(block_list), 1);

fprintf('\n==================== DIM100 Block Sweep Start ====================\n');
fprintf('N_train=%d, N_val=%d, Reduction=PLS-15, Order=3, q=0.8\n', Benchmark.N_train, Benchmark.N_val);

for i = 1:numel(block_list)
    num_blocks = block_list(i);
    fprintf('\n[%d/%d] GEAPCK_BLOCK num_blocks=%d\n', i, numel(block_list), num_blocks);
    try
        opts = ModelOpts;
        opts.GEBlockOpts.num_blocks = num_blocks;

        t_train = tic;
        [results, opts_out, ~, ReduceModel] = aPC_Train(data, opts);
        train_sec = toc(t_train);

        t_pred = tic;
        y_pred = aPC_Predict(data.X_val, results, opts_out, ReduceModel);
        pred_sec = toc(t_pred);

        relmse = sum((y_pred(:) - data.Y_val(:)).^2) / sum((data.Y_val(:) - mean(data.Y_val(:))).^2);
        loo = results.LOO;

        bench(i).NumBlocks = num_blocks;
        bench(i).TrainSec = train_sec;
        bench(i).PredictSec = pred_sec;
        bench(i).TotalSec = train_sec + pred_sec;
        bench(i).RelMSE = relmse;
        bench(i).LOO = loo;
        bench(i).Status = 'OK';

        fprintf('  RelMSE = %.6e | Train = %.3fs | Pred = %.3fs | LOO = %.6e\n', relmse, train_sec, pred_sec, loo);
    catch ME
        bench(i).NumBlocks = num_blocks;
        bench(i).Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

fprintf('\n==================== DIM100 Block Sweep Summary ====================\n');
fprintf('%-10s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
    'Blocks', 'Train(s)', 'Pred(s)', 'Total(s)', 'RelMSE', 'LOO', 'Status');
fprintf('%s\n', repmat('-', 1, 96));
for i = 1:numel(bench)
    fprintf('%-10d | %10s | %10s | %10s | %12s | %12s | %s\n', ...
        bench(i).NumBlocks, fmt_num(bench(i).TrainSec), fmt_num(bench(i).PredictSec), ...
        fmt_num(bench(i).TotalSec), fmt_num(bench(i).RelMSE), fmt_num(bench(i).LOO), bench(i).Status);
end
fprintf('%s\n', repmat('=', 1, 62));

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
