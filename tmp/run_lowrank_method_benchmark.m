clear; clc;
rng(80);

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'core'));
aPC;
uqlab;

%% User config
case_name = 'wingweight_nored';
methods = {'hybrid_residual_subspace'};

% Shared low-rank options
lowrank_rank = [];
residual_rank = 'auto';
operator_eta_grid = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0];
operator_grad_weight = 0.1;

%% Build benchmark case
[data, base, case_title] = buildCase(case_name);

if isempty(lowrank_rank)
    lowrank_rank = defaultRank(case_name);
end

fprintf('\n==================== LOW-RANK METHOD BENCHMARK ====================\n');
fprintf('Case: %s\n', case_title);
fprintf('Methods: %s\n', strjoin(methods, ', '));
fprintf('Rank: %d\n', lowrank_rank);

bench = repmat(struct( ...
    'Method', '', 'TrainSec', NaN, 'PredSec', NaN, 'RelMSE', NaN, ...
    'LOO', NaN, 'Status', '', 'Extra', ''), numel(methods), 1);

for i = 1:numel(methods)
    opts = base;
    opts.ModelType = 'geapck_lowrank';
    opts.GELowRankOpts.rank = lowrank_rank;
    opts.GELowRankOpts.method = methods{i};
    if ~isempty(residual_rank)
        opts.GELowRankOpts.residual_rank = residual_rank;
    end
    if strcmpi(methods{i}, 'supervised_joint_operator')
        opts.GELowRankOpts.operator_eta_grid = operator_eta_grid;
        opts.GELowRankOpts.operator_grad_weight = operator_grad_weight;
    end

    fprintf('\n[%d/%d] %s\n', i, numel(methods), upper(methods{i}));
    bench(i) = runOne(data, opts, methods{i});
end

fprintf('\n%-28s | %10s | %10s | %12s | %12s | %-24s | %s\n', ...
    'Method', 'Train(s)', 'Pred(s)', 'RelMSE', 'LOO', 'Extra', 'Status');
fprintf('%s\n', repmat('-', 1, 132));
for i = 1:numel(bench)
    fprintf('%-28s | %10s | %10s | %12s | %12s | %-24s | %s\n', ...
        bench(i).Method, fmtNum(bench(i).TrainSec), fmtNum(bench(i).PredSec), ...
        fmtNum(bench(i).RelMSE), fmtNum(bench(i).LOO), bench(i).Extra, bench(i).Status);
end

function [data, base, case_title] = buildCase(case_name)
    switch lower(case_name)
        case 'dim100_pls12'
            myInput = uq_createInput(dim100_Input());
            data = struct();
            data.X_train = uq_getSample(myInput, 200);
            data.Y_train = dim100(data.X_train);
            data.X_val = uq_getSample(myInput, 500);
            data.Y_val = dim100(data.X_val);
            data.ModelFcn = @dim100;

            base = struct();
            base.Order = 3;
            base.qnorm = 0.8;
            base.aPC_Method = 'stieltjes';
            base.RegressionMethod = 'lars';
            base.TrendBuilder = 'lars';
            base.PCKType = 'optimal';
            base.Parallel = 0;
            base.ReductionMethod = 'pls';
            base.red_dim = 12;
            base.GradientMode = 'finite-difference';
            base.KrigingOpts.corr = 'gaussian';
            base.KrigingOpts.theta0 = 1.0;
            base.KrigingOpts.theta_lb = 1e-3;
            base.KrigingOpts.theta_ub = 1e2;
            base.KrigingOpts.nugget = 1e-8;
            base.KrigingOpts.trend_range = [0.5, 1.0];
            base.KrigingOpts.optimize_theta = true;
            base.KrigingOpts.DisplayProgress = false;
            base.FDOpts.type = 'central';
            base.FDOpts.step = 1e-6;
            case_title = 'DIM100 | PLS-12 | finite-difference gradients';

        case 'dim150_mcp12'
            myInput = uq_createInput(dim150_Input());
            data = struct();
            data.X_train = uq_getSample(myInput, 200);
            data.Y_train = dim150(data.X_train);
            data.X_val = uq_getSample(myInput, 1000);
            data.Y_val = dim150(data.X_val);
            data.ModelFcn = @dim150;

            base = struct();
            base.Order = 3;
            base.qnorm = 0.8;
            base.aPC_Method = 'stieltjes';
            base.RegressionMethod = 'lars';
            base.TrendBuilder = 'lars';
            base.PCKType = 'optimal';
            base.Parallel = 0;
            base.ReductionMethod = 'mcp';
            base.red_dim = 12;
            base.GradientMode = 'finite-difference';
            base.KrigingOpts.corr = 'gaussian';
            base.KrigingOpts.theta0 = 1.0;
            base.KrigingOpts.theta_lb = 1e-3;
            base.KrigingOpts.theta_ub = 1e2;
            base.KrigingOpts.nugget = 1e-8;
            base.KrigingOpts.trend_range = [0.2, 0.8];
            base.KrigingOpts.optimize_theta = true;
            base.KrigingOpts.DisplayProgress = false;
            base.FDOpts.type = 'central';
            base.FDOpts.step = 1e-6;
            case_title = 'DIM150 | MCP-12 | finite-difference gradients';

        case 'wingweight_nored'
            myInput = uq_createInput(wingweight_Input());
            data = struct();
            data.X_train = uq_getSample(myInput, 200);
            data.Y_train = wingweight(data.X_train);
            data.dYdX_train = wingweight_grad(data.X_train);
            data.X_val = uq_getSample(myInput, 300);
            data.Y_val = wingweight(data.X_val);
            data.ModelFcn = @wingweight;

            base = struct();
            base.Order = 4;
            base.qnorm = 0.8;
            base.aPC_Method = 'stieltjes';
            base.RegressionMethod = 'lars';
            base.TrendBuilder = 'lars';
            base.PCKType = 'optimal';
            base.Parallel = 0;
            base.GradientMode = 'provided';
            base.KrigingOpts.corr = 'gaussian';
            base.KrigingOpts.theta0 = 1.0;
            base.KrigingOpts.theta_lb = 1e-3;
            base.KrigingOpts.theta_ub = 1e2;
            base.KrigingOpts.nugget = 1e-8;
            base.KrigingOpts.trend_range = [0.2, 0.8];
            base.KrigingOpts.optimize_theta = true;
            base.KrigingOpts.DisplayProgress = false;
            case_title = 'WINGWEIGHT | no reduction | analytic gradients';

        case 'morris10_pls8'
            myInput = uq_createInput(morris10_Input());
            data = struct();
            data.X_train = uq_getSample(myInput, 200);
            data.Y_train = morris10(data.X_train);
            data.dYdX_train = morris10_grad(data.X_train);
            data.X_val = uq_getSample(myInput, 1000);
            data.Y_val = morris10(data.X_val);
            data.ModelFcn = @morris10;

            base = struct();
            base.Order = 4;
            base.qnorm = 0.9;
            base.aPC_Method = 'stieltjes';
            base.RegressionMethod = 'lars';
            base.TrendBuilder = 'lars';
            base.PCKType = 'optimal';
            base.Parallel = 0;
            base.ReductionMethod = 'pls';
            base.red_dim = 8;
            base.GradientMode = 'provided';
            base.KrigingOpts.corr = 'gaussian';
            base.KrigingOpts.theta0 = 1.0;
            base.KrigingOpts.theta_lb = 1e-3;
            base.KrigingOpts.theta_ub = 1e2;
            base.KrigingOpts.nugget = 1e-8;
            base.KrigingOpts.trend_range = [0.2, 0.8];
            base.KrigingOpts.optimize_theta = true;
            base.KrigingOpts.DisplayProgress = false;
            case_title = 'MORRIS10 | PLS-8 | analytic gradients';

        otherwise
            error('Unknown case_name: %s', case_name);
    end
end

function rank = defaultRank(case_name)
    switch lower(case_name)
        case 'dim100_pls12'
            rank = 3;
        case 'dim150_mcp12'
            rank = 9;
        case 'wingweight_nored'
            rank = 4;
        case 'morris10_pls8'
            rank = 4;
        otherwise
            rank = 3;
    end
end

function out = runOne(data, opts, method_name)
    out = struct('Method', upper(method_name), 'TrainSec', NaN, 'PredSec', NaN, ...
        'RelMSE', NaN, 'LOO', NaN, 'Status', '', 'Extra', '');
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
        out.Extra = buildExtra(results);

        fprintf('  RelMSE = %.6e | LOO = %.6e | Train = %.3fs | Pred = %.3fs', ...
            out.RelMSE, out.LOO, out.TrainSec, out.PredSec);
        if ~strcmp(out.Extra, '-')
            fprintf(' | %s', out.Extra);
        end
        fprintf('\n');
    catch ME
        out.Status = ['FAIL: ' ME.identifier];
        fprintf(2, '  FAILED: %s\n', ME.message);
    end
end

function extra = buildExtra(results)
    extra = '-';
    if ~isfield(results, 'PCKModel') || ~isfield(results.PCKModel, 'lowrank')
        return;
    end
    lowrank = results.PCKModel.lowrank;
    parts = {sprintf('rank=%d', lowrank.rank)};

    if isfield(lowrank, 'main_rank') && isfield(lowrank, 'residual_rank')
        parts{end + 1} = sprintf('main=%d,res=%d', lowrank.main_rank, lowrank.residual_rank);
    end
    if isfield(lowrank, 'operator_selected_eta')
        parts{end + 1} = sprintf('eta=%.2f', lowrank.operator_selected_eta);
    end

    extra = strjoin(parts, ' | ');
end

function s = fmtNum(x)
    if isempty(x) || ~isfinite(x)
        s = '-';
    elseif abs(x) >= 1e3 || abs(x) < 1e-2
        s = sprintf('%.3e', x);
    else
        s = sprintf('%.4f', x);
    end
end
