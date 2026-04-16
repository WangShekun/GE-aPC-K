clear; clc;
rng(80);

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'core'));
aPC;
uqlab;

C = localCaseWingweightNoReduction();

configs = {
    struct('name', 'apck', 'apply', @localAsAPCK), ...
    struct('name', 'geapck', 'apply', @localAsGE), ...
    struct('name', 'geapck_lowrank_r2', 'apply', @(opts) localAsLowRank(opts, 2)), ...
    struct('name', 'geapck_lowrank_r3', 'apply', @(opts) localAsLowRank(opts, 3)) ...
};

fprintf('\n==================== GE Low-Rank Wingweight No-Reduction ====================\n');
fprintf('\n---- Case: %s ----\n', upper(C.name));
bench = repmat(struct('Name', '', 'TrainSec', NaN, 'PredSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', ''), numel(configs), 1);
for i = 1:numel(configs)
    cfg = configs{i};
    fprintf('[%d/%d] %s\n', i, numel(configs), upper(cfg.name));
    bench(i) = localRunOne(C.data, cfg.name, cfg.apply(C.baseOpts));
end
localPrintBench(bench);

function C = localCaseWingweightNoReduction()
    myInput = uq_createInput(wingweight_Input());
    data = struct();
    data.X_train = uq_getSample(myInput, 80);
    data.Y_train = wingweight(data.X_train);
    data.dYdX_train = wingweight_grad(data.X_train);
    data.X_val = uq_getSample(myInput, 800);
    data.Y_val = wingweight(data.X_val);

    opts = struct();
    opts.Order = 2;
    opts.qnorm = 0.8;
    opts.aPC_Method = 'stieltjes';
    opts.RegressionMethod = 'lars';
    opts.TrendBuilder = 'lars';
    opts.PCKType = 'optimal';
    opts.GradientMode = 'provided';
    opts.Parallel = 0;
    opts.KrigingOpts.corr = 'gaussian';
    opts.KrigingOpts.theta0 = 1.0;
    opts.KrigingOpts.theta_lb = 1e-3;
    opts.KrigingOpts.theta_ub = 1e2;
    opts.KrigingOpts.nugget = 1e-8;
    opts.KrigingOpts.trend_range = [0.5, 1.0];
    opts.KrigingOpts.optimize_theta = true;
    opts.KrigingOpts.DisplayProgress = false;

    C = struct('name', 'wingweight_nored', 'data', data, 'baseOpts', opts);
end

function opts = localAsAPCK(opts)
    opts.ModelType = 'apck';
end

function opts = localAsGE(opts)
    opts.ModelType = 'geapck';
end

function opts = localAsLowRank(opts, rank)
    opts.ModelType = 'geapck_lowrank';
    opts.GELowRankOpts.rank = rank;
    opts.GELowRankOpts.method = 'svd';
end

function out = localRunOne(data, name, opts)
    out = struct('Name', upper(name), 'TrainSec', NaN, 'PredSec', NaN, 'RelMSE', NaN, 'LOO', NaN, 'Status', '');
    try
        tTrain = tic;
        [results, opts_out, ~, ReduceModel] = aPC_Train(data, opts);
        out.TrainSec = toc(tTrain);

        tPred = tic;
        y_pred = aPC_Predict(data.X_val, results, opts_out, ReduceModel);
        out.PredSec = toc(tPred);

        out.RelMSE = sum((y_pred(:) - data.Y_val(:)).^2) / sum((data.Y_val(:) - mean(data.Y_val(:))).^2);
        out.LOO = results.LOO;
        out.Status = 'OK';
        fprintf('  RelMSE = %.6e | LOO = %.6e | Train = %.3fs | Pred = %.3fs\n', out.RelMSE, out.LOO, out.TrainSec, out.PredSec);
    catch ME
        out.Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

function localPrintBench(bench)
    fprintf('\n%-20s | %10s | %10s | %12s | %12s | %s\n', 'Model', 'Train(s)', 'Pred(s)', 'RelMSE', 'LOO', 'Status');
    fprintf('%s\n', repmat('-', 1, 96));
    for i = 1:numel(bench)
        fprintf('%-20s | %10s | %10s | %12s | %12s | %s\n', ...
            bench(i).Name, localFmtNum(bench(i).TrainSec), localFmtNum(bench(i).PredSec), ...
            localFmtNum(bench(i).RelMSE), localFmtNum(bench(i).LOO), bench(i).Status);
    end
end

function s = localFmtNum(x)
    if isempty(x) || ~isfinite(x)
        s = '-';
    elseif abs(x) >= 1e3 || abs(x) < 1e-2
        s = sprintf('%.3e', x);
    else
        s = sprintf('%.4f', x);
    end
end
