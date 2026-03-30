function tests = TestModelTypes
tests = functiontests(localfunctions);
end

function setupOnce(~)
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'core'));
aPC;
rng(11);
end

function testMissingModelTypeErrors(testCase)
data = makeData(16, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.Parallel = 0;

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingModelType');
end

function testAPCKSequentialRuns(testCase)
data = makeData(24, 3);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'sequential';
ModelOpts.KrigingOpts.trend_range = [0.2, 0.8];
ModelOpts.Parallel = 0;

[results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, lower(results.ModelType), 'apck');
verifyEqual(testCase, lower(results.PCKType), 'sequential');
verifyTrue(testCase, isempty(ReduceModel));
verifyGreaterThanOrEqual(testCase, numel(results.PCKModel.trend_idx), 1);
verifyEqual(testCase, sort(results.PCKModel.trend_idx), sort(results.TrendPathIdx(:)'));

Yhat = aPC_Predict(data.X_train, results, ModelOptsOut, ReduceModel);
verifySize(testCase, Yhat, [size(data.X_train,1), 1]);
end

function testGEAPCKProvidedAnalyticGradientRuns(testCase)
    N = 18;
    X = rand(N, 2) * 2 - 1;
    Y = exp(0.3 * X(:,1)) + 0.5 * X(:,2).^2;
    dYdX = [0.3 * exp(0.3 * X(:,1)), X(:,2)];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck';
    ModelOpts.PCKType = 'optimal';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);

    verifyEqual(testCase, lower(results.ModelType), 'geapck');
    verifyEqual(testCase, lower(results.PCKType), 'optimal');
    verifyTrue(testCase, isfield(dataOut, 'dYdX_train'));
    verifySize(testCase, dataOut.dYdX_train, [N, 2]);
    verifyLessThan(testCase, max(abs(dataOut.dYdX_train(:) - dYdX(:))), 1e-12);

    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);
    verifySize(testCase, Yhat, [N, 1]);
end

function testAPCKThetaOptimizationUsesFmincon(testCase)
data = makeData(22, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.theta0 = 0.3;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, results.PCKModel.theta_optimizer, 'fmincon');
verifyGreaterThanOrEqual(testCase, min(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_lb);
verifyLessThanOrEqual(testCase, max(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_ub);
end

function testAPCKThetaOptimizationKeepsFiniteDiagnosticsAfterRefactor(testCase)
rng(11);
data = makeData(22, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.theta0 = 0.5;
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfinite(results.LOO));
verifyTrue(testCase, isfinite(results.normEmpErr));
verifyTrue(testCase, all(isfinite(results.PCKModel.theta)));
verifyTrue(testCase, isfinite(results.PCKModel.loo));
verifyTrue(testCase, isfinite(results.PCKModel.normEmpErr));
end

function testAPCKThetaOptimizationReportsObjectiveGradientUsage(testCase)
rng(11);
data = makeData(22, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.use_objective_gradient = true;
ModelOpts.KrigingOpts.theta0 = 0.3;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, results.PCKModel.theta_uses_objective_gradient);
verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
end

function testAPCKThetaObjectiveGradientKeepsFiniteDiagnostics(testCase)
rng(11);
data = makeData(22, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.use_objective_gradient = true;
ModelOpts.KrigingOpts.theta0 = 0.5;
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfinite(results.LOO));
verifyTrue(testCase, isfinite(results.normEmpErr));
verifyTrue(testCase, all(isfinite(results.PCKModel.theta)));
verifyTrue(testCase, isfinite(results.PCKModel.loo));
verifyTrue(testCase, isfinite(results.PCKModel.normEmpErr));
verifyGreaterThanOrEqual(testCase, min(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_lb);
verifyLessThanOrEqual(testCase, max(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_ub);
end

function testAPCKThetaCoreBatchesLinearSolves(testCase)
rng(11);
data = makeData(18, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.theta0 = 0.5;
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

profile clear;
profile on;
[~, ~, ~, ~] = aPC_Train(data, ModelOpts);
p = profile('info');
profile off;

idx = find(strcmp({p.FunctionTable.FunctionName}, 'aPC_PCKBackend>localEvalThetaCore'), 1);
verifyNotEmpty(testCase, idx);

lines = p.FunctionTable(idx).ExecutedLines(:, 1);
merged_line = localFindSourceLine('coefficients/aPC_PCKBackend.m', 'KinvHY = L'' \ (L \ [H, y_aug]);');
alpha_line = localFindSourceLine('coefficients/aPC_PCKBackend.m', 'alpha = L'' \ (L \ residual);');

verifyNotEmpty(testCase, merged_line);
verifyTrue(testCase, any(lines == merged_line));
if ~isempty(alpha_line)
    verifyFalse(testCase, any(lines == alpha_line));
end
end

function testAPCKCholeskyFailureDoesNotLookLikePerfectLOO(testCase)
X = [0 0; 0 0; 0.5 -0.2; -0.5 0.3];
Y = sin(X(:,1)) + X(:,2).^2;

data = struct();
data.X_train = X;
data.Y_train = Y;

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.nugget = 0;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isinf(results.LOO));
verifyTrue(testCase, isinf(results.normEmpErr));
verifyTrue(testCase, isinf(results.PCKModel.loo));
verifyTrue(testCase, isinf(results.PCKModel.normEmpErr));
verifyFalse(testCase, isempty(results.PCKModel.trend_idx));
verifyEqual(testCase, results.coefficients(results.PCKModel.trend_idx), ...
    zeros(numel(results.PCKModel.trend_idx), 1), 'AbsTol', 1e-12);
end

function testAPCKSerialOptimalWarmStartsThetaAfterFirstTrend(testCase)
rng(11);
data = makeData(22, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.theta0 = 0.3;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyGreaterThan(testCase, numel(results.PCKModel.trend_idx), 1);
verifyTrue(testCase, isfield(results.PCKModel, 'theta_init_used'));
verifyEqual(testCase, numel(results.PCKModel.theta_init_used), numel(results.PCKModel.theta));
verifyFalse(testCase, isempty(results.PCKModel.theta_init_used));
end

function testGEAPCKCorrelationVectorizationPreservesOutputs(testCase)
rng(11);
N = 10;
X = rand(N, 2) * 2 - 1;
Y = exp(0.2 * X(:,1)) + X(:,2).^2;
dYdX = [0.2 * exp(0.2 * X(:,1)), 2 * X(:,2)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifySize(testCase, Yhat, [N, 1]);
verifyLessThan(testCase, results.LOO, 2e-3);
verifyLessThan(testCase, max(abs(Yhat - Y)), 3e-2);
verifySize(testCase, dataOut.dYdX_train, [N, 2]);
end

function testPckBuildAugmentedSystemAvoidsCell2mat(testCase)
U = [0.1 -0.3; -0.2 0.4; 0.5 0.2];
Y = [1.0; 0.7; -0.4];
G = [0.3 -0.1; 0.2 0.4; -0.5 0.6];
F = [ones(3, 1), U(:, 1)];
dF = [zeros(3, 1), ones(3, 1); zeros(3, 1), zeros(3, 1)];
theta = [0.8 1.1];

profile clear;
profile on;
[K, y_aug, H] = pckBuildAugmentedSystem(U, Y, G, F, dF, theta, 'gaussian', 1e-8); %#ok<ASGLU>
p = profile('info');
profile off;

names = {p.FunctionTable.FunctionName};
verifyFalse(testCase, any(strcmp(names, 'cell2mat')));
verifySize(testCase, K, [9, 9]);
verifySize(testCase, y_aug, [9, 1]);
verifySize(testCase, H, [9, 2]);
end

function testGEAPCKStoresDerivativeValueCache(testCase)
rng(11);
N = 10;
X = rand(N, 2) * 2 - 1;
Y = exp(0.2 * X(:,1)) + X(:,2).^2;
dYdX = [0.2 * exp(0.2 * X(:,1)), 2 * X(:,2)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfield(results.basis_info, 'univ_dvals'));
verifySize(testCase, results.basis_info.univ_dvals, size(results.basis_info.univ_vals));
end

function testGEAPCKGaussianUsesObjectiveGradient(testCase)
rng(11);
N = 12;
X = rand(N, 2) * 2 - 1;
Y = exp(0.2 * X(:,1)) + X(:,2).^2;
dYdX = [0.2 * exp(0.2 * X(:,1)), 2 * X(:,2)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.corr = 'gaussian';
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.use_objective_gradient = true;
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, results.PCKModel.theta_uses_objective_gradient);
verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
verifyTrue(testCase, isfinite(results.PCKModel.loo));
end

function testGEAPCKMatern32SkipsObjectiveGradient(testCase)
rng(11);
N = 12;
X = rand(N, 2) * 2 - 1;
Y = exp(0.2 * X(:,1)) + X(:,2).^2;
dYdX = [0.2 * exp(0.2 * X(:,1)), 2 * X(:,2)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.corr = 'matern32';
ModelOpts.KrigingOpts.optimize_theta = true;
ModelOpts.KrigingOpts.use_objective_gradient = true;
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyFalse(testCase, results.PCKModel.theta_uses_objective_gradient);
verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
verifyTrue(testCase, isfinite(results.PCKModel.loo));
end

function testAPCKMatern32Runs(testCase)
data = makeData(24, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.corr = 'matern32';
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.1];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(data.X_train, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.PCKModel.corr), 'matern32');
verifySize(testCase, Yhat, [size(data.X_train, 1), 1]);
end

function testGEAPCKMatern32Runs(testCase)
N = 14;
X = rand(N, 2) * 2 - 1;
Y = exp(0.2 * X(:,1)) + X(:,2).^2;
dYdX = [0.2 * exp(0.2 * X(:,1)), 2 * X(:,2)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.corr = 'matern32';
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.9, 0.6];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.PCKModel.corr), 'matern32');
verifySize(testCase, Yhat, [N, 1]);
verifyLessThan(testCase, max(abs(Yhat - Y)), 5e-2);
end

function testGEAPCKSupportsSPCAHSICGradientPropagation(testCase)
N = 20;
X = rand(N, 3) * 2 - 1;
Y = sin(X(:,1)) + 0.5 * X(:,2).^2 - 0.2 * X(:,3);
dYdX = [cos(X(:,1)), X(:,2), -0.2 * ones(N, 1)];

data = struct();
data.X_train = X;
data.Y_train = Y;
data.dYdX_train = dYdX;

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'geapck';
ModelOpts.PCKType = 'optimal';
ModelOpts.GradientMode = 'provided';
ModelOpts.ReductionMethod = 'spca_hsic';
ModelOpts.red_dim = 2;
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyFalse(testCase, isempty(ReduceModel));
verifySize(testCase, dataOut.dY_model, [N, 2]);
verifySize(testCase, Yhat, [N, 1]);
end

function testSearchFieldsDoNotIncludeTheta0(testCase)
fields = aPC_list_search_fields();
verifyFalse(testCase, any(strcmp(fields, 'KrigingOpts.theta0')));
end

function testParallelAPCKPrintPathDoesNotUseUndefinedMo(testCase)
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = [2, 3];
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 1;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(data.X_train, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'apck');
verifySize(testCase, Yhat, [size(data.X_train, 1), 1]);
end

function testOptimalAPCKSingleConfigUsesTrendParallelScope(testCase)
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 1;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, results.PCKModel.parallel_scope, 'trend');
end

function testTrendParallelPrintsCorrectScopeLabel(testCase)
rng(11);
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 1;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

out = evalc('[results, ~, ~, ~] = aPC_Train(data, ModelOpts);');

verifyEqual(testCase, lower(results.PCKModel.parallel_scope), 'trend');
verifyNotEmpty(testCase, strfind(out, '(Trend-Parallel)')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Parallel]')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'trend 1/2')); %#ok<STREMP>
verifyEmpty(testCase, strfind(out, '[Trend-Serial]')); %#ok<STREMP>
end

function testParallelOuterPrintsUnifiedHeaderAndFooter(testCase)
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = [2, 3];
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 1;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

out = evalc('[~, ~, ~, ~] = aPC_Train(data, ModelOpts);');

verifyNotEmpty(testCase, strfind(out, 'aPC TRAIN BEGIN')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '(Parallel) Evaluating 2 configuration(s)...')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'aPC TRAIN END')); %#ok<STREMP>
end

function testSerialProgressShowsStepCounts(testCase)
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = [2, 3];
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.Parallel = 0;

out = evalc('[~, ~, ~, ~] = aPC_Train(data, ModelOpts);');

verifyNotEmpty(testCase, strfind(out, '(1/2)')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '(2/2)')); %#ok<STREMP>
end

function testSerialOptimalPCKShowsTrendProgress(testCase)
rng(11);
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.DisplayProgress = true;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

out = evalc('[~, ~, ~, ~] = aPC_Train(data, ModelOpts);');

verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 1/2')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 2/2')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[###############---------------]')); %#ok<STREMP>
end

function testSerialOptimalPCKCanStopAfterConsecutiveNonImprovements(testCase)
rng(1);
data = makeData(24, 3);

ModelOpts = struct();
ModelOpts.Order = 4;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.DisplayProgress = true;
ModelOpts.KrigingOpts.trend_range = [1, 6];
ModelOpts.KrigingOpts.early_stop_patience = 2;

out = evalc('[~, ~, ~, ~] = aPC_Train(data, ModelOpts);');

verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 1/5')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 2/5')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 3/5')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 4/5')); %#ok<STREMP>
verifyEmpty(testCase, strfind(out, '[aPC_PCKBackend] [Trend-Serial] trend 5/5')); %#ok<STREMP>
end

function testSensitivityBlockedForAPCK(testCase)
data = makeData(24, 3);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.KrigingOpts.trend_range = [0.2, 0.8];
ModelOpts.Parallel = 0;

[results, ModelOptsOut, ~, ~] = aPC_Train(data, ModelOpts);

verifyError(testCase, @() aPC_ComputeSensitivity(results, ModelOptsOut), ...
    'aPC:SensitivityOnlyForAPC');
end

function testAPCPrintDefaultReportIsStructured(testCase)
[results, ModelOpts, data] = makePrintFixture('apck', false);

out = evalc('aPC_print(results, ModelOpts, data);');

verifyNotEmpty(testCase, strfind(out, 'aPC REPORT')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Model Summary')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Data Summary')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Error Summary')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'PCK type')); %#ok<STREMP>
verifyEmpty(testCase, strfind(out, 'Reduction Summary')); %#ok<STREMP>
verifyEmpty(testCase, strfind(out, 'Orthogonality')); %#ok<STREMP>
verifyEmpty(testCase, strfind(out, 'PCE Coefficients')); %#ok<STREMP>
end

function testAPCPrintShowsReductionSummaryConditionally(testCase)
[results, ModelOpts, data] = makePrintFixture('geapck', true);

out = evalc('aPC_print(results, ModelOpts, data);');

verifyNotEmpty(testCase, strfind(out, 'Reduction Summary')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Dimension reduction')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Reduced dimension')); %#ok<STREMP>
end

function testAPCKPrintShowsTrendBuilderAndSelectedTrendCount(testCase)
rng(11);
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ModelOptsOut, dataOut, ~] = aPC_Train(data, ModelOpts);
out = evalc('aPC_print(results, ModelOptsOut, dataOut);');

verifyEmpty(testCase, strfind(out, 'Regression method')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Trend builder                : LARS')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, sprintf('Active terms                 : %d', numel(results.PCKModel.trend_idx)))); %#ok<STREMP>
end

function testAPCKStoresTrendCoefficientsInResultsVector(testCase)
rng(11);
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

expected = zeros(size(results.coefficients));
expected(results.PCKModel.trend_idx) = results.PCKModel.beta;

verifyEqual(testCase, results.coefficients, expected, 'AbsTol', 1e-12);
verifyGreaterThan(testCase, nnz(results.coefficients), 0);
end

function testAPCKRejectsUnsupportedTrendBuilder(testCase)
rng(11);
data = makeData(20, 2);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.TrendBuilder = 'omp';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.Parallel = 0;

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:UnsupportedTrendBuilder');
end

function testAPCPrintUQIsStructured(testCase)
UQ = struct('mean', 1.2, 'var', 0.3, 'std', sqrt(0.3), 'CI95', [0.1 2.3], 'VaR95', 0.2);
out = evalc('aPC_printUQ(UQ);');

verifyNotEmpty(testCase, strfind(out, 'UQ REPORT')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Core Statistics')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Extended Statistics')); %#ok<STREMP>
end

function testAPCPrintSensitivityIsStructured(testCase)
SA = struct();
SA.S_first = [0.2 0.3 0.1];
SA.S_total = [0.4 0.5 0.2];
SA.S_second = zeros(3, 3);
SA.S_second(1, 2) = 0.05;
SA.S_second(1, 3) = 0.02;
SA.S_second(2, 3) = 0.01;
ModelOpts = struct('num_vars', 3);

out = evalc('aPC_printSensitivity(SA, ModelOpts);');

verifyNotEmpty(testCase, strfind(out, 'SENSITIVITY REPORT')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Total-Order Indices')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'First-Order Indices')); %#ok<STREMP>
verifyNotEmpty(testCase, strfind(out, 'Second-Order Indices')); %#ok<STREMP>
end

function data = makeData(N, d)
X = rand(N, d) * 2 - 1;
Y = sin(X(:,1));
if d >= 2
    Y = Y + X(:,2).^2;
end
if d >= 3
    Y = Y - 0.5 * X(:,3);
end

data = struct();
data.X_train = X;
data.Y_train = Y;
end

function [results, ModelOpts, data] = makePrintFixture(modelType, withReduction)
results = struct();
results.RegressionMethod = 'LARS';
results.ModelType = lower(modelType);
results.PCKType = 'optimal';
results.coefficients = [1; 0; 2];
results.optErrorParams = struct('loo', 1.23e-3);
results.RelMSE = 4.56e-4;
results.N_train = 20;
results.N_test = 10;
results.Indices = [0 0; 1 0; 0 1];

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 0.7;
ModelOpts.num_vars = 2;
ModelOpts.Indices = results.Indices;
if withReduction
    ModelOpts.ReductionMethod = 'pls';
    ModelOpts.red_dim = 1;
end

data = struct();
data.X_train = rand(20, 2);
data.Y_train = rand(20, 1);
end

function line_num = localFindSourceLine(rel_path, needle)
repo_root = fileparts(fileparts(mfilename('fullpath')));
txt = fileread(fullfile(repo_root, rel_path));
parts = regexp(txt, '\r\n|\n|\r', 'split');
line_num = find(cellfun(@(s) contains(s, needle), parts), 1);
end
