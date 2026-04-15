clear; clc;
rng(80);

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'core'));
aPC;
uqlab;

baseOpts = struct();
baseOpts.Order = 4;
baseOpts.qnorm = 0.9;
baseOpts.aPC_Method = 'stieltjes';
baseOpts.RegressionMethod = 'lars';
baseOpts.TrendBuilder = 'lars';
baseOpts.PCKType = 'optimal';
baseOpts.Parallel = 0;

baseOpts.KrigingOpts.corr = 'gaussian';
baseOpts.KrigingOpts.theta0 = 1.0;
baseOpts.KrigingOpts.theta_lb = 1e-3;
baseOpts.KrigingOpts.theta_ub = 1e2;
baseOpts.KrigingOpts.nugget = 1e-8;
baseOpts.KrigingOpts.trend_range = [0.2 0.8];
baseOpts.KrigingOpts.optimize_theta = true;
baseOpts.KrigingOpts.DisplayProgress = true;

Benchmark = struct();
Benchmark.N_train = 200;
Benchmark.N_val = 2000;

benchSpec = get_benchmark_spec('wingweight');
myInput = uq_createInput(benchSpec.InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, Benchmark.N_train);
data.Y_train = benchSpec.ModelFcn(data.X_train);
data.dYdX_train = benchSpec.GradFcn(data.X_train);
data.ModelFcn = benchSpec.ModelFcn;
data.X_val = uq_getSample(myInput, Benchmark.N_val);
data.Y_val = benchSpec.ModelFcn(data.X_val);

rows = 6;
bench = repmat(struct('Name', '', 'TrainSec', NaN, 'PredictSec', NaN, ...
    'TotalSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', ''), rows, 1);

fprintf('\n==================== WINGWEIGHT Compare Start ====================\n');
fprintf('N_train=%d, N_val=%d, No Reduction, Order=4, q=0.9\n', Benchmark.N_train, Benchmark.N_val);

model_names = {'apc', 'apck'};
row_idx = 1;
for i = 1:numel(model_names)
    name = model_names{i};
    fprintf('\n[%d/%d] %s\n', row_idx, rows, upper(name));
    bench(row_idx) = run_one(data, baseOpts, name, []);
    row_idx = row_idx + 1;
end

for num_blocks = 2:5
    label = sprintf('geapck_block(nb=%d)', num_blocks);
    fprintf('\n[%d/%d] %s\n', row_idx, rows, upper(label));
    bench(row_idx) = run_one(data, baseOpts, 'geapck_block', num_blocks);
    row_idx = row_idx + 1;
end

fprintf('\n==================== WINGWEIGHT Compare Summary ====================\n');
fprintf('%-20s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
    'Model', 'Train(s)', 'Pred(s)', 'Total(s)', 'RelMSE', 'LOO', 'Status');
fprintf('%s\n', repmat('-', 1, 110));
for i = 1:numel(bench)
    fprintf('%-20s | %10s | %10s | %10s | %12s | %12s | %s\n', ...
        bench(i).Name, fmt_num(bench(i).TrainSec), fmt_num(bench(i).PredictSec), ...
        fmt_num(bench(i).TotalSec), fmt_num(bench(i).RelMSE), fmt_num(bench(i).LOO), bench(i).Status);
end
fprintf('%s\n', repmat('=', 1, 74));

function out = run_one(data, baseOpts, model_name, num_blocks)
out = struct('Name', upper(model_name), 'TrainSec', NaN, 'PredictSec', NaN, ...
    'TotalSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', '');
try
    opts = baseOpts;
    opts.ModelType = lower(model_name);
    if strcmpi(opts.ModelType, 'apc')
        if isfield(opts, 'PCKType')
            opts = rmfield(opts, 'PCKType');
        end
    elseif strcmpi(opts.ModelType, 'geapck_block')
        opts.GradientMode = 'provided';
        opts.GEBlockOpts.num_blocks = num_blocks;
        out.Name = sprintf('GEAPCK_BLOCK(%d)', num_blocks);
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

    relmse = sum((y_pred(:) - data.Y_val(:)).^2) / sum((data.Y_val(:) - mean(data.Y_val(:))).^2);
    out.TrainSec = train_sec;
    out.PredictSec = pred_sec;
    out.TotalSec = train_sec + pred_sec;
    out.RelMSE = relmse;
    out.LOO = loo;
    out.Status = 'OK';
    fprintf('  RelMSE = %.6e | Train = %.3fs | Pred = %.3fs | LOO = %.6e\n', relmse, train_sec, pred_sec, loo);
catch ME
    out.Status = ['FAIL: ' ME.identifier];
    fprintf(2, '  FAILED: %s\n', ME.message);
end
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
    case 'wingweight'
        spec.InputOpts = wingweight_Input;
        spec.ModelFcn = @wingweight;
        spec.GradFcn = @wingweight_grad;
    otherwise
        error('Unsupported benchmark: %s', name);
end
end
