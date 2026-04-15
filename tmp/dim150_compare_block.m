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
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.ReductionMethod = 'pls';
ModelOpts.red_dim = 20;

ModelOpts.KrigingOpts.corr = 'gaussian';
ModelOpts.KrigingOpts.theta0 = 1.0;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 1e2;
ModelOpts.KrigingOpts.nugget = 1e-8;
ModelOpts.KrigingOpts.trend_range = [0.8];
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.DisplayProgress = true;

Benchmark = struct();
Benchmark.N_train = 200;
Benchmark.N_val = 2000;
Benchmark.FDType = 'central';
Benchmark.FDStep = 1e-6;

myInput = uq_createInput(dim150_Input);
data = struct();
data.X_train = uq_getSample(myInput, Benchmark.N_train);
data.Y_train = dim150(data.X_train);
data.ModelFcn = @dim150;
data.X_val = uq_getSample(myInput, Benchmark.N_val);
data.Y_val = dim150(data.X_val);

model_list = {'apc', 'apck', 'geapck_block'};
bench = repmat(struct('Name', '', 'TrainSec', NaN, 'PredictSec', NaN, ...
    'TotalSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', ''), numel(model_list), 1);

fprintf('\n==================== DIM150 Compare Start ====================\n');
fprintf('N_train=%d, N_val=%d, Reduction=PLS-20, Order=3, q=0.8\n', Benchmark.N_train, Benchmark.N_val);

for i = 1:numel(model_list)
    name = model_list{i};
    fprintf('\n[%d/%d] %s\n', i, numel(model_list), upper(name));
    try
        opts = ModelOpts;
        opts.ModelType = name;

        if strcmpi(name, 'apc')
            if isfield(opts, 'PCKType')
                opts = rmfield(opts, 'PCKType');
            end
        end

        if strcmpi(name, 'geapck') || strcmpi(name, 'geapck_block')
            opts.GradientMode = 'finite-difference';
            opts.FDOpts.type = Benchmark.FDType;
            opts.FDOpts.step = Benchmark.FDStep;
        end

        t_train = tic;
        [results, opts_out, data_out, ReduceModel] = aPC_Train(data, opts);
        train_sec = toc(t_train);

        t_pred = tic;
        y_pred = aPC_Predict(data.X_val, results, opts_out, ReduceModel);
        pred_sec = toc(t_pred);

        relmse = sum((y_pred(:) - data.Y_val(:)).^2) / sum((data.Y_val(:) - mean(data.Y_val(:))).^2);
        loo = NaN;
        if isfield(results, 'LOO')
            loo = results.LOO;
        elseif isfield(results, 'optErrorParams') && isfield(results.optErrorParams, 'loo')
            loo = results.optErrorParams.loo;
        end

        bench(i).Name = upper(name);
        bench(i).TrainSec = train_sec;
        bench(i).PredictSec = pred_sec;
        bench(i).TotalSec = train_sec + pred_sec;
        bench(i).RelMSE = relmse;
        bench(i).LOO = loo;
        bench(i).Status = 'OK';

        fprintf('  RelMSE = %.6e | Train = %.3fs | Pred = %.3fs | LOO = %.6e\n', relmse, train_sec, pred_sec, loo);
    catch ME
        bench(i).Name = upper(name);
        bench(i).Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

fprintf('\n==================== DIM150 Compare Summary ====================\n');
fprintf('%-13s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
    'Model', 'Train(s)', 'Pred(s)', 'Total(s)', 'RelMSE', 'LOO', 'Status');
fprintf('%s\n', repmat('-', 1, 102));
for i = 1:numel(bench)
    fprintf('%-13s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
        bench(i).Name, fmt_num(bench(i).TrainSec), fmt_num(bench(i).PredictSec), ...
        fmt_num(bench(i).TotalSec), fmt_num(bench(i).RelMSE), fmt_num(bench(i).LOO), bench(i).Status);
end
fprintf('%s\n', repmat('=', 1, 68));

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
