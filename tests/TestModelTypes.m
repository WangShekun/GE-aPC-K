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
