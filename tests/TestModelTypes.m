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

function testGEAPCKSequentialThetaOptimizationRuns(testCase)
    rng(17);
    N = 14;
    X = rand(N, 2) * 2 - 1;
    Y = exp(0.25 * X(:,1)) + 0.4 * X(:,2).^2;
    dYdX = [0.25 * exp(0.25 * X(:,1)), 0.8 * X(:,2)];

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
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.KrigingOpts.corr = 'gaussian';
    ModelOpts.KrigingOpts.optimize_theta = true;
    ModelOpts.KrigingOpts.theta0 = [0.6, 0.9];
    ModelOpts.KrigingOpts.theta_lb = 1e-2;
    ModelOpts.KrigingOpts.theta_ub = 10;

    [results, ~, ~, ~] = aPC_Train(data, ModelOpts);

    verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
    verifyTrue(testCase, all(isfinite(results.PCKModel.theta)));
end

function testGEAPCKLowRankProvidedAnalyticGradientRuns(testCase)
    rng(33);
    N = 22;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.2 * X(:,1)) + 0.4 * X(:,2).^2 + 0.3 * X(:,1) .* X(:,3);
    dYdX = [0.2 * exp(0.2 * X(:,1)) + 0.3 * X(:,3), 0.8 * X(:,2), 0.3 * X(:,1)];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 2;
    ModelOpts.GELowRankOpts.method = 'svd';

    [results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.ModelType), 'geapck_lowrank');
    verifyEqual(testCase, lower(results.PCKType), 'sequential');
    verifyTrue(testCase, isfield(dataOut, 'dYdX_train'));
    verifyTrue(testCase, isfield(results.PCKModel, 'lowrank'));
    verifyEqual(testCase, results.PCKModel.lowrank.rank, 2);
    verifySize(testCase, results.PCKModel.lowrank.V_r, [3, 2]);
    verifySize(testCase, results.PCKModel.lowrank.Z, [N, 2]);
    verifySize(testCase, Yhat, [N, 1]);
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testGEAPCKLowRankHybridResidualSubspaceRuns(testCase)
    rng(43);
    N = 24;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.2 * X(:,1)) + 0.6 * X(:,1) .* X(:,2) + 0.2 * X(:,3).^3;
    dYdX = [0.2 * exp(0.2 * X(:,1)) + 0.6 * X(:,2), 0.6 * X(:,1), 0.6 * X(:,3).^2];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 2;
    ModelOpts.GELowRankOpts.method = 'hybrid_residual_subspace';

    [results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.PCKModel.lowrank_options.method), 'hybrid_residual_subspace');
    verifyEqual(testCase, results.PCKModel.lowrank.rank, 2);
    verifyEqual(testCase, lower(results.PCKModel.lowrank.hybrid_residual_mode), 'post_project');
    verifySize(testCase, results.PCKModel.lowrank.V_r, [3, 2]);
    verifySize(testCase, results.PCKModel.lowrank.Z, [N, 2]);
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.singular_values)));
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testGEAPCKLowRankHybridResidualSubspaceProjectedModeRuns(testCase)
    rng(44);
    N = 24;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.2 * X(:,1)) + 0.6 * X(:,1) .* X(:,2) + 0.2 * X(:,3).^3;
    dYdX = [0.2 * exp(0.2 * X(:,1)) + 0.6 * X(:,2), 0.6 * X(:,1), 0.6 * X(:,3).^2];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 2;
    ModelOpts.GELowRankOpts.method = 'hybrid_residual_subspace';
    ModelOpts.GELowRankOpts.hybrid_residual_mode = 'project_then_svd';

    [results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.PCKModel.lowrank.hybrid_residual_mode), 'project_then_svd');
    verifyEqual(testCase, results.PCKModel.lowrank.rank, 2);
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testGEAPCKLowRankHybridResidualSubspaceAutoResidualRankRuns(testCase)
    rng(71);
    N = 24;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.2 * X(:,1)) + 0.6 * X(:,1) .* X(:,2) + 0.2 * X(:,3).^3;
    dYdX = [0.2 * exp(0.2 * X(:,1)) + 0.6 * X(:,2), 0.6 * X(:,1), 0.6 * X(:,3).^2];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 3;
    ModelOpts.GELowRankOpts.method = 'hybrid_residual_subspace';
    ModelOpts.GELowRankOpts.residual_rank = 'auto';

    [results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.PCKModel.lowrank_options.method), 'hybrid_residual_subspace');
    verifyEqual(testCase, results.PCKModel.lowrank.rank, 3);
    verifyTrue(testCase, isfield(results.PCKModel.lowrank, 'auto_residual_selected_rank'));
    verifyTrue(testCase, numel(results.PCKModel.lowrank.auto_residual_rank_grid) >= 1);
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.auto_residual_proxy_scores)));
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testGEAPCKLowRankSupervisedJointOperatorRuns(testCase)
    rng(61);
    N = 26;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.2 * X(:,1)) + 0.6 * X(:,1) .* X(:,2) + 0.2 * X(:,3).^3;
    dYdX = [0.2 * exp(0.2 * X(:,1)) + 0.6 * X(:,2), 0.6 * X(:,1), 0.6 * X(:,3).^2];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 2;
    ModelOpts.GELowRankOpts.method = 'supervised_joint_operator';
    ModelOpts.GELowRankOpts.operator_eta_grid = [0, 0.5, 1.0];

    [results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.PCKModel.lowrank_options.method), 'supervised_joint_operator');
    verifyEqual(testCase, results.PCKModel.lowrank.rank, 2);
    verifyTrue(testCase, isfield(results.PCKModel.lowrank, 'operator_selected_eta'));
    verifyEqual(testCase, numel(results.PCKModel.lowrank.operator_eta_grid), 3);
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.operator_proxy_scores)));
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.operator_value_scores)));
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.operator_grad_scores)));
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testGEAPCKLowRankSupervisedJointOperatorFminbndRuns(testCase)
    rng(67);
    N = 24;
    X = rand(N, 3) * 2 - 1;
    Y = exp(0.18 * X(:,1)) + 0.45 * X(:,1) .* X(:,2) + 0.12 * X(:,3).^3;
    dYdX = [0.18 * exp(0.18 * X(:,1)) + 0.45 * X(:,2), 0.45 * X(:,1), 0.36 * X(:,3).^2];

    data = struct();
    data.X_train = X;
    data.Y_train = Y;
    data.dYdX_train = dYdX;

    ModelOpts = struct();
    ModelOpts.Order = 3;
    ModelOpts.qnorm = 1.0;
    ModelOpts.aPC_Method = 'stieltjes';
    ModelOpts.RegressionMethod = 'lars';
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.PCKType = 'sequential';
    ModelOpts.GradientMode = 'provided';
    ModelOpts.Parallel = 0;
    ModelOpts.GELowRankOpts.rank = 2;
    ModelOpts.GELowRankOpts.method = 'supervised_joint_operator';
    ModelOpts.GELowRankOpts.operator_search_method = 'fminbnd';
    ModelOpts.GELowRankOpts.operator_eta_grid = [0, 0.5, 1.0];

    [results, ModelOptsOut, ~, ReduceModel] = aPC_Train(data, ModelOpts);
    Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

    verifyEqual(testCase, lower(results.PCKModel.lowrank.operator_search_method), 'fminbnd');
    verifyTrue(testCase, isfield(results.PCKModel.lowrank, 'operator_sample_eta'));
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.operator_sample_eta)));
    verifyTrue(testCase, all(isfinite(results.PCKModel.lowrank.operator_proxy_scores)));
    verifyTrue(testCase, all(isfinite(Yhat)));
end

function testAutoSearchIncludesGELowRankRank(testCase)
    ModelOpts = struct();
    ModelOpts.ModelType = 'geapck_lowrank';
    ModelOpts.Order = 2;
    ModelOpts.qnorm = 0.8;
    ModelOpts.GELowRankOpts.rank = [2 3 5];
    ModelOpts.GELowRankOpts.method = 'svd';
    ModelOpts.KrigingOpts.corr = 'gaussian';

    grid = aPC_build_search_grid(ModelOpts);

    verifyNumElements(testCase, grid, 3);
    verifyEqual(testCase, arrayfun(@(g) g.GELowRankOpts.rank, grid), [2 3 5]);
    verifyTrue(testCase, all(arrayfun(@(g) strcmp(g.GELowRankOpts.method, 'svd'), grid)));
    verifyTrue(testCase, all(arrayfun(@(g) strcmp(g.ModelType, 'geapck_lowrank'), grid)));
end

function testGEAPCKLowRankGaussianAssemblyMatchesLegacyProjection(testCase)
    fixture = localMakeLowRankAssemblyFixture();

    [K_new, y_new, H_new, lowrank_new] = pckBuildLowRankAugmentedSystem( ...
        fixture.U, fixture.Y, fixture.G, fixture.F, fixture.dF, ...
        fixture.theta, 'gaussian', fixture.nugget, fixture.rank);
    [K_ref, y_ref, H_ref, lowrank_ref] = localBuildLegacyLowRankAugmentedSystem( ...
        fixture.U, fixture.Y, fixture.G, fixture.F, fixture.dF, ...
        fixture.theta, 'gaussian', fixture.nugget, fixture.rank);

    verifyEqual(testCase, K_new, K_ref, 'AbsTol', 1e-12);
    verifyEqual(testCase, y_new, y_ref, 'AbsTol', 1e-12);
    verifyEqual(testCase, H_new, H_ref, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.rank, lowrank_ref.rank);
    verifyEqual(testCase, lowrank_new.V_r, lowrank_ref.V_r, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.Z, lowrank_ref.Z, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.singular_values, lowrank_ref.singular_values, 'AbsTol', 1e-12);
end

function testGEAPCKLowRankMatern32AssemblyMatchesLegacyProjection(testCase)
    fixture = localMakeLowRankAssemblyFixture();

    [K_new, y_new, H_new, lowrank_new] = pckBuildLowRankAugmentedSystem( ...
        fixture.U, fixture.Y, fixture.G, fixture.F, fixture.dF, ...
        fixture.theta, 'matern32', fixture.nugget, fixture.rank);
    [K_ref, y_ref, H_ref, lowrank_ref] = localBuildLegacyLowRankAugmentedSystem( ...
        fixture.U, fixture.Y, fixture.G, fixture.F, fixture.dF, ...
        fixture.theta, 'matern32', fixture.nugget, fixture.rank);

    verifyEqual(testCase, K_new, K_ref, 'AbsTol', 1e-11);
    verifyEqual(testCase, y_new, y_ref, 'AbsTol', 1e-12);
    verifyEqual(testCase, H_new, H_ref, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.rank, lowrank_ref.rank);
    verifyEqual(testCase, lowrank_new.V_r, lowrank_ref.V_r, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.Z, lowrank_ref.Z, 'AbsTol', 1e-12);
    verifyEqual(testCase, lowrank_new.singular_values, lowrank_ref.singular_values, 'AbsTol', 1e-12);
end


function testGEAPCKLowRankGaussianPredictionMatchesLegacyProjection(testCase)
    fixture = localMakeLowRankPredictionFixture();

    y_new = aPC_PCKPredictLowRank(fixture.Psi_new, fixture.U_new, fixture.PCKModel);
    y_ref = localPredictLowRankLegacy(fixture.Psi_new, fixture.U_new, fixture.PCKModel);

    verifyEqual(testCase, y_new, y_ref, 'AbsTol', 1e-12);
end

function testGEAPCKLowRankMatern32PredictionMatchesLegacyProjection(testCase)
    fixture = localMakeLowRankPredictionFixture();
    fixture.PCKModel.corr = 'matern32';

    y_new = aPC_PCKPredictLowRank(fixture.Psi_new, fixture.U_new, fixture.PCKModel);
    y_ref = localPredictLowRankLegacy(fixture.Psi_new, fixture.U_new, fixture.PCKModel);

    verifyEqual(testCase, y_new, y_ref, 'AbsTol', 1e-11);
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

function testAPCKThetaOptimizationHidesObjectiveGradientDiagnostics(testCase)
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
ModelOpts.KrigingOpts.theta0 = 0.3;
ModelOpts.KrigingOpts.theta_lb = 1e-3;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.4];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
verifyFalse(testCase, isfield(results.PCKModel, 'theta_uses_objective_gradient'));
end

function testAPCKThetaOptimizationKeepsFiniteDiagnostics(testCase)
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

function testGEGaussianThetaOptHidesObjectiveGradientDiagnostics(testCase)
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
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
verifyFalse(testCase, isfield(results.PCKModel, 'theta_uses_objective_gradient'));
verifyTrue(testCase, isfinite(results.PCKModel.loo));
end

function testGEMatern32ThetaOptHidesObjectiveGradientDiagnostics(testCase)
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
ModelOpts.KrigingOpts.theta0 = [0.7, 1.1];
ModelOpts.KrigingOpts.theta_lb = 1e-2;
ModelOpts.KrigingOpts.theta_ub = 10;
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ~, ~, ~] = aPC_Train(data, ModelOpts);

verifyTrue(testCase, isfield(results.PCKModel, 'theta_opt_output'));
verifyGreaterThan(testCase, results.PCKModel.theta_opt_output.funcCount, 0);
verifyFalse(testCase, isfield(results.PCKModel, 'theta_uses_objective_gradient'));
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
verifyEqual(testCase, size(dataOut.dY_model, 1), N);
verifyEqual(testCase, size(dataOut.dY_model, 2), ReduceModel.params.ncomp_used);
verifyLessThanOrEqual(testCase, ReduceModel.params.ncomp_used, ModelOpts.red_dim);
verifySize(testCase, Yhat, [N, 1]);
end

function testPrepareModelGradientsRejectsUnknownReductionMethod(testCase)
data = struct();
data.X_train = [1 2 3; 4 5 6];
data.dYdX_train = [0.1 0.2 0.3; 0.4 0.5 0.6];

map_U = struct();
map_U.nonConstIdx = [true true true];
map_U.sigma = [2 3 4];

ReduceModel = struct();
ReduceModel.method = 'mystery-linear-map';
ReduceModel.params = struct('V', eye(3, 2));
ReduceModel.sigma2 = [5 7];

ModelOpts = struct();
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_PrepareModelGradients(data, map_U, ReduceModel, ModelOpts), ...
    'aPC:UnsupportedGEReduction');
end

function testPrepareModelGradientsAllowsOnlyPrecomputedUGradient(testCase)
data = struct();
data.X_train = [1 2 3; 4 5 6];
data.dYdU_train = [0.11 0.22 0.33; 0.44 0.55 0.66];

map_U = struct();
map_U.nonConstIdx = [true true true];
map_U.sigma = [2 3 4];

ModelOpts = struct();
ModelOpts.GradientMode = 'provided';

[dY_model, dYdX] = aPC_PrepareModelGradients(data, map_U, [], ModelOpts);

verifyEqual(testCase, dY_model, data.dYdU_train, 'AbsTol', 1e-12);
verifyTrue(testCase, isempty(dYdX));
end

function testSPCAHSICModelInfoDoesNotStoreKy(testCase)
U = rand(16, 4);
Y = sin(U(:,1)) + U(:,2).^2;

[~, ReduceModel] = aPC_dimReduction_spca_hsic(U, Y, 2);

verifyTrue(testCase, isfield(ReduceModel, 'info'));
verifyFalse(testCase, isfield(ReduceModel.info, 'Ky'));
end

function testGEPLSReductionRequiresGradients(testCase)
data = struct();
data.X_train = rand(8, 3);
data.Y_train = rand(8, 1);

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.ReductionMethod = 'gepls';
ModelOpts.red_dim = 2;
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingProvidedGradient');
end

function testASPLSReductionRequiresGradients(testCase)
data = struct();
data.X_train = rand(8, 3);
data.Y_train = rand(8, 1);

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.ReductionMethod = 'as_pls';
ModelOpts.red_dim = 2;
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingProvidedGradient');
end

function testASReductionRequiresGradients(testCase)
data = struct();
data.X_train = rand(8, 3);
data.Y_train = rand(8, 1);

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.ReductionMethod = 'as';
ModelOpts.red_dim = 2;
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingProvidedGradient');
end

function testSASReductionRequiresGradients(testCase)
data = struct();
data.X_train = rand(8, 3);
data.Y_train = rand(8, 1);

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.ReductionMethod = 'sas';
ModelOpts.red_dim = 2;
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingProvidedGradient');
end

function testSGSReductionRequiresGradients(testCase)
data = struct();
data.X_train = rand(8, 3);
data.Y_train = rand(8, 1);

ModelOpts = struct();
ModelOpts.Order = 2;
ModelOpts.qnorm = 1.0;
ModelOpts.aPC_Method = 'stieltjes';
ModelOpts.RegressionMethod = 'lars';
ModelOpts.ModelType = 'apc';
ModelOpts.ReductionMethod = 'sgs';
ModelOpts.red_dim = 2;
ModelOpts.GradientMode = 'provided';

verifyError(testCase, @() aPC_Train(data, ModelOpts), 'aPC:MissingProvidedGradient');
end

function testGEAPCKSupportsGEPLSGradientPropagation(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'gepls';
ModelOpts.red_dim = 2;
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyEqual(testCase, lower(ReduceModel.method), 'gepls');
verifySize(testCase, dataOut.dY_model, [N, 2]);
verifySize(testCase, Yhat, [N, 1]);
end

function testGEAPCKSupportsASPLSGradientPropagation(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'as_pls';
ModelOpts.red_dim = 2;
ModelOpts.as_pls_opts = struct('as_dim', 3);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyEqual(testCase, lower(ReduceModel.method), 'as_pls');
verifySize(testCase, dataOut.dY_model, [N, 2]);
verifySize(testCase, Yhat, [N, 1]);
end

function testGEAPCKSupportsASGradientPropagation(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'as';
ModelOpts.red_dim = 2;
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyEqual(testCase, lower(ReduceModel.method), 'as');
verifyFalse(testCase, isfield(ReduceModel, 'info'));
verifySize(testCase, dataOut.dY_model, [N, 2]);
verifySize(testCase, Yhat, [N, 1]);
end

function testASReductionStandaloneDoesNotRequireModelOptsOrInfo(testCase)
U = rand(10, 4);
dYdU = rand(10, 4);

[X_red, ReduceModel] = aPC_dimReduction_as(U, dYdU, 2);

verifySize(testCase, X_red, [10, 2]);
verifyEqual(testCase, lower(ReduceModel.method), 'as');
verifyFalse(testCase, isfield(ReduceModel, 'info'));
verifyTrue(testCase, isa(ReduceModel.transform, 'function_handle'));
end

function testGEAPCKSupportsSASGradientPropagation(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'sas';
ModelOpts.red_dim = 2;
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyEqual(testCase, lower(ReduceModel.method), 'sas');
verifyEqual(testCase, size(dataOut.dY_model, 1), N);
verifyEqual(testCase, size(dataOut.dY_model, 2), size(ReduceModel.params.A, 2));
verifyEqual(testCase, size(ReduceModel.params.A, 2), ModelOpts.red_dim);
verifySize(testCase, Yhat, [N, 1]);
end

function testGEAPCKSupportsSGSGradientPropagation(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'sgs';
ModelOpts.red_dim = 2;
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[results, ModelOptsOut, dataOut, ReduceModel] = aPC_Train(data, ModelOpts);
Yhat = aPC_Predict(X, results, ModelOptsOut, ReduceModel);

verifyEqual(testCase, lower(results.ModelType), 'geapck');
verifyEqual(testCase, lower(ReduceModel.method), 'sgs');
verifySize(testCase, dataOut.dY_model, [N, 2]);
verifySize(testCase, Yhat, [N, 1]);
end

function testGEPLSReductionUsesNestedOpts(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'gepls';
ModelOpts.red_dim = 2;
ModelOpts.gepls_opts = struct('gamma', 2.5);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[~, ~, ~, ReduceModel] = aPC_Train(data, ModelOpts);
verifyEqual(testCase, ReduceModel.params.gamma, 2.5, 'AbsTol', 1e-12);
end

function testASPLSReductionUsesNestedOpts(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'as_pls';
ModelOpts.red_dim = 2;
ModelOpts.as_pls_opts = struct('as_dim', 2);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[~, ~, ~, ReduceModel] = aPC_Train(data, ModelOpts);
verifyEqual(testCase, ReduceModel.params.as_dim, 2);
end

function testSASReductionUsesNestedOpts(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'sas';
ModelOpts.red_dim = 2;
ModelOpts.sas_opts = struct('response_power', 3.0, 'weight_eps', 1e-6);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[~, ~, ~, ReduceModel] = aPC_Train(data, ModelOpts);
verifyEqual(testCase, ReduceModel.params.response_power, 3.0, 'AbsTol', 1e-12);
verifyEqual(testCase, ReduceModel.params.weight_eps, 1e-6, 'AbsTol', 1e-12);
verifyEqual(testCase, lower(ReduceModel.params.weight_method), 'zpower');
verifyEqual(testCase, size(ReduceModel.params.A, 2), ModelOpts.red_dim);
end

function testSASReductionSupportsDensityWeightMethod(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'sas';
ModelOpts.red_dim = 2;
ModelOpts.sas_opts = struct('weight_method', 'density', 'gamma', 0.7, 'sigma_y', 0.25);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[~, ~, ~, ReduceModel] = aPC_Train(data, ModelOpts);

verifyEqual(testCase, lower(ReduceModel.params.weight_method), 'density');
verifyEqual(testCase, ReduceModel.params.gamma, 0.7, 'AbsTol', 1e-12);
verifyGreaterThanOrEqual(testCase, min(ReduceModel.params.weights), 1.0 - 1e-12);
verifyLessThanOrEqual(testCase, max(ReduceModel.params.weights), 1.7 + 1e-12);
end

function testSGSReductionUsesNestedOpts(testCase)
N = 18;
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
ModelOpts.ReductionMethod = 'sgs';
ModelOpts.red_dim = 2;
ModelOpts.sgs_opts = struct('alpha', 0.9);
ModelOpts.Parallel = 0;
ModelOpts.KrigingOpts.optimize_theta = false;
ModelOpts.KrigingOpts.theta0 = [0.8, 1.2];
ModelOpts.KrigingOpts.trend_range = [0.2, 0.2];

[~, ~, ~, ReduceModel] = aPC_Train(data, ModelOpts);
verifyEqual(testCase, ReduceModel.params.alpha, 0.9, 'AbsTol', 1e-12);
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

function fixture = localMakeLowRankAssemblyFixture()
fixture.U = [-0.8 0.2 0.5; -0.1 -0.4 0.7; 0.3 0.6 -0.2; 0.9 -0.7 0.1; 0.4 0.1 -0.6];
fixture.Y = [0.7; -0.1; 0.5; 1.2; -0.4];
fixture.G = [0.8 -0.2 0.1; 0.3 0.5 -0.4; -0.6 0.7 0.2; 0.4 -0.1 0.9; -0.2 0.3 0.6];
fixture.F = [ones(5, 1), fixture.U(:, 1), fixture.U(:, 2) .* fixture.U(:, 3)];
fixture.dF = [ ...
    zeros(5, 1), ones(5, 1), zeros(5, 1); ...
    zeros(5, 1), zeros(5, 1), fixture.U(:, 3); ...
    zeros(5, 1), zeros(5, 1), fixture.U(:, 2) ...
    ];
fixture.theta = [0.9, 1.3, 0.7];
fixture.nugget = 1e-8;
fixture.rank = 2;
end

function [K, y_aug, H, lowrank] = localBuildLegacyLowRankAugmentedSystem(U, Y, G, F, dF, theta, corr_type, nugget, rank)
N = size(U, 1);
d = size(U, 2);
r = min(round(rank), d);

[~, Sg, Vg] = svd(G, 'econ');
V_r = Vg(:, 1:r);
Z = G * V_r;
[Kff, Kfg_blocks, Kgg_blocks] = pckEvalKernel(U, U, theta, corr_type);

Kfg_modes = localProjectKfgLegacy(Kfg_blocks, V_r);
Kgg_modes = localProjectKggLegacy(Kgg_blocks, V_r);

nobs = N * (r + 1);
K = zeros(nobs, nobs);
K(1:N, 1:N) = Kff;
for k = 1:r
    rows = N + (k - 1) * N + (1:N);
    K(1:N, rows) = Kfg_modes{k};
    K(rows, 1:N) = -Kfg_modes{k};
    for l = 1:r
        cols = N + (l - 1) * N + (1:N);
        K(rows, cols) = Kgg_modes{k, l};
    end
end

K(1:nobs + 1:end) = K(1:nobs + 1:end) + nugget;
y_aug = [Y; reshape(Z, N * r, 1)];
H = [F; kron(V_r', eye(N)) * dF];

lowrank = struct();
lowrank.rank = r;
lowrank.V_r = V_r;
lowrank.Z = Z;
lowrank.singular_values = diag(Sg);
end

function fixture = localMakeLowRankPredictionFixture()
fixture = struct();
fixture.U_new = [-0.6 0.5 -0.1; 0.2 -0.3 0.8; 0.7 0.4 -0.5];
fixture.Psi_new = [ ...
    ones(3, 1), fixture.U_new(:, 1), fixture.U_new(:, 2), fixture.U_new(:, 3), ...
    fixture.U_new(:, 1) .* fixture.U_new(:, 2) ...
    ];

fixture.PCKModel = struct();
fixture.PCKModel.trend_idx = [1 2 5];
fixture.PCKModel.beta = [0.8; -0.4; 0.25];
fixture.PCKModel.U_train = [-0.8 0.2 0.5; -0.1 -0.4 0.7; 0.3 0.6 -0.2; 0.9 -0.7 0.1; 0.4 0.1 -0.6];
fixture.PCKModel.theta = [0.9, 1.3, 0.7];
fixture.PCKModel.corr = 'gaussian';
fixture.PCKModel.alpha = [0.3; -0.2; 0.5; 0.1; -0.4; 0.2; -0.3; 0.1; 0.6; -0.2; 0.4; -0.5; 0.2; 0.1; -0.1];
fixture.PCKModel.lowrank = struct('V_r', [0.8 -0.2; 0.3 0.9; -0.5 0.4]);
end

function y_pred = localPredictLowRankLegacy(Psi_new, U_new, PCKModel)
trend_idx = PCKModel.trend_idx;
Fnew = Psi_new(:, trend_idx);
cov_vec = localLowRankCrossCovarianceLegacy(U_new, PCKModel.U_train, PCKModel.theta, PCKModel.corr, PCKModel.lowrank.V_r);
y_pred = Fnew * PCKModel.beta + cov_vec * PCKModel.alpha;
end

function cov_vec = localLowRankCrossCovarianceLegacy(U_new, U_train, theta, corr_type, V_r)
[Kff, Kfg_blocks] = pckEvalKernel(U_new, U_train, theta, corr_type);
Kfg_modes = localProjectKfgLegacy(Kfg_blocks, V_r);
cov_vec = [Kff, cat(2, Kfg_modes{:})];
end

function Kfg_modes = localProjectKfgLegacy(Kfg_blocks, V_r)
d = size(V_r, 1);
r = size(V_r, 2);
Kfg_modes = cell(1, r);
for k = 1:r
    acc = zeros(size(Kfg_blocks{1}));
    for a = 1:d
        acc = acc + V_r(a, k) * Kfg_blocks{a};
    end
    Kfg_modes{k} = acc;
end
end

function Kgg_modes = localProjectKggLegacy(Kgg_blocks, V_r)
d = size(V_r, 1);
r = size(V_r, 2);
Kgg_modes = cell(r, r);
base = zeros(size(Kgg_blocks{1, 1}));
for k = 1:r
    for l = 1:r
        acc = base;
        for a = 1:d
            for b = 1:d
                acc = acc + V_r(a, k) * V_r(b, l) * Kgg_blocks{a, b};
            end
        end
        Kgg_modes{k, l} = acc;
    end
end
end
