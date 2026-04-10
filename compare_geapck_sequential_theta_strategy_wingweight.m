clear; clc; close all;

rng(80);

repo_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, 'core'));
aPC
uqlab

Benchmark = struct();
Benchmark.Name = 'wingweight';
Benchmark.N_train = 200;
Benchmark.N_val = 2000;

base_opts = struct();
base_opts.Order = 5;
base_opts.qnorm = 0.8;
base_opts.aPC_Method = 'stieltjes';
base_opts.RegressionMethod = 'lars';
base_opts.TrendBuilder = 'lars';
base_opts.ModelType = 'geapck';
base_opts.PCKType = 'sequential';
base_opts.Parallel = 0;
% base_opts.ReductionMethod = 'sas';
% base_opts.red_dim = 20;
% base_opts.sas_opts.weight_method = 'zpower';
base_opts.GradientMode = 'finite-difference';
base_opts.FDOpts.type = 'central';
base_opts.FDOpts.step = 1e-6;
base_opts.KrigingOpts.corr = 'gaussian';
base_opts.KrigingOpts.theta0 = 1.0;
base_opts.KrigingOpts.theta_lb = 1e-3;
base_opts.KrigingOpts.theta_ub = 1e2;
base_opts.KrigingOpts.nugget = 1e-8;
base_opts.KrigingOpts.optimize_theta = true;
base_opts.KrigingOpts.DisplayProgress = false;
base_opts.KrigingOpts.theta_refine_max_iterations = 40;
base_opts.KrigingOpts.theta_refine_max_function_evaluations = 60;

bench_spec = localGetBenchmarkSpec(Benchmark.Name);
myInput = uq_createInput(bench_spec.InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, Benchmark.N_train);
data.Y_train = bench_spec.ModelFcn(data.X_train);
data.ModelFcn = bench_spec.ModelFcn;
data.X_val = uq_getSample(myInput, Benchmark.N_val);
data.Y_val = bench_spec.ModelFcn(data.X_val);

cases = {
    'direct', 'Direct GE theta';
    'apck_surrogate', 'APCK surrogate';
    'apck_surrogate_refine', 'APCK surrogate + refine';
    };

rows = size(cases, 1);
bench = repmat(struct( ...
    'Label', '', ...
    'TrainSec', NaN, ...
    'PredictSec', NaN, ...
    'TotalSec', NaN, ...
    'RelMSE', NaN, ...
    'LOO', NaN, ...
    'Theta', [], ...
    'Status', ''), rows, 1);

fprintf('\n==================== GEAPCK Sequential Theta Strategy Compare ====================\n');
fprintf('Benchmark: %s | p = %d | q = %.2f | RD = NONE | Grad = %s | N_train = %d | N_val = %d\n', ...
    upper(Benchmark.Name), base_opts.Order, base_opts.qnorm, upper(base_opts.GradientMode), ...
    Benchmark.N_train, Benchmark.N_val);

for i = 1:rows
    strategy = cases{i, 1};
    label = cases{i, 2};
    fprintf('\n[%d/%d] %s\n', i, rows, label);
    try
        bench(i) = localRunCase(data, base_opts, strategy, label);
        bench(i).Status = 'OK';
        fprintf('  RelMSE = %.6e | Train = %.3fs | Pred = %.3fs | LOO = %.6e\n', ...
            bench(i).RelMSE, bench(i).TrainSec, bench(i).PredictSec, bench(i).LOO);
    catch ME
        bench(i).Label = label;
        bench(i).Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

fprintf('\n==================== Summary ====================\n');
fprintf('%-26s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
    'Strategy', 'Train(s)', 'Pred(s)', 'Total(s)', 'RelMSE', 'LOO', 'Status');
fprintf('%s\n', repmat('-', 1, 108));
for i = 1:rows
    fprintf('%-26s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
        bench(i).Label, localFmtNum(bench(i).TrainSec), localFmtNum(bench(i).PredictSec), ...
        localFmtNum(bench(i).TotalSec), localFmtNum(bench(i).RelMSE), ...
        localFmtNum(bench(i).LOO), bench(i).Status);
end

baseline = bench(1);
fprintf('\n==================== Relative To Direct ====================\n');
for i = 2:rows
    if ~strcmp(bench(i).Status, 'OK') || ~strcmp(baseline.Status, 'OK')
        continue;
    end
    fprintf('%s\n', bench(i).Label);
    fprintf('  Train speedup      : %.3fx\n', baseline.TrainSec / bench(i).TrainSec);
    fprintf('  Total speedup      : %.3fx\n', baseline.TotalSec / bench(i).TotalSec);
    fprintf('  RelMSE ratio       : %.3fx\n', bench(i).RelMSE / baseline.RelMSE);
    fprintf('  LOO ratio          : %.3fx\n', bench(i).LOO / baseline.LOO);
    fprintf('  Final theta        : [%s]\n', num2str(bench(i).Theta, ' %.4g'));
end

fprintf('\nDirect final theta   : [%s]\n', num2str(baseline.Theta, ' %.4g'));

function out = localRunCase(data, base_opts, theta_strategy, label)
    rng(80);
    opts = base_opts;
    opts.KrigingOpts.theta_strategy = theta_strategy;

    t_train = tic;
    [results, opts_out, ~, ReduceModel] = aPC_Train(data, opts);
    train_sec = toc(t_train);

    t_pred = tic;
    y_pred = aPC_Predict(data.X_val, results, opts_out, ReduceModel);
    pred_sec = toc(t_pred);

    out = struct();
    out.Label = label;
    out.TrainSec = train_sec;
    out.PredictSec = pred_sec;
    out.TotalSec = train_sec + pred_sec;
    out.RelMSE = mean((y_pred - data.Y_val).^2) / max(var(data.Y_val, 1), eps);
    out.LOO = results.LOO;
    out.Theta = results.PCKModel.theta;
    out.Status = 'OK';
end

function spec = localGetBenchmarkSpec(name)
    switch lower(strtrim(name))
        case 'wingweight'
            spec.InputOpts = wingweight_Input;
            spec.ModelFcn = @wingweight;
            spec.GradFcn = @wingweight_grad;

        case 'morris10'
            spec.InputOpts = morris10_Input;
            spec.ModelFcn = @morris10;
            spec.GradFcn = @morris10_grad;

        case 'dim100'
            spec.InputOpts = dim100_Input;
            spec.ModelFcn = @dim100;
            spec.GradFcn = @dim100_grad;
        otherwise
            error('Unsupported benchmark: %s', name);
    end
end

function s = localFmtNum(x)
    if isempty(x) || any(isnan(x(:)))
        s = 'NaN';
    else
        s = sprintf('%.6g', x);
    end
end
