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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.8];
ModelOpts.Parallel = 0;

[results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, lower(results.ModelType), 'apck');
verifyEqual(testCase, lower(results.PCKType), 'sequential');
verifyTrue(testCase, isempty(ReduceModel));
verifyGreaterThanOrEqual(testCase, numel(results.PCKModel.trend_idx), 1);

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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, results.PCKModel.theta_optimizer, 'fmincon');
verifyGreaterThanOrEqual(testCase, min(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_lb);
verifyLessThanOrEqual(testCase, max(results.PCKModel.theta), ModelOpts.KrigingOpts.theta_ub);
end

function testGEAPCKCorrelationVectorizationPreservesOutputs(testCase)
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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifySize(testCase, Yhat, [N, 1]);
verifyLessThan(testCase, results.LOO, 1e-3);
verifyLessThan(testCase, max(abs(Yhat - Y)), 3e-2);
verifySize(testCase, dataOut.dYdX_train, [N, 2]);
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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.4];

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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.2];

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
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.2];

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

function testSensitivityBlockedForAPCK(testCase)
data = makeData(24, 3);

ModelOpts = struct();
ModelOpts.Order = 3;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apck';
ModelOpts.PCKType = 'optimal';
ModelOpts.KrigingOpts.max_trend_terms = [0.2, 0.8];
ModelOpts.Parallel = 0;

[results, ModelOptsOut, ~, ~] = aPC_Train(data, ModelOpts);

verifyError(testCase, @() aPC_ComputeSensitivity(results, ModelOptsOut), ...
    'aPC:SensitivityOnlyForAPC');
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
