function candidate = aPC_TrainCandidate(data, map_U, ModelOpts)
% Train one candidate configuration in the current model space.

    [data_model, data_grad, ModelOptsOut, ReduceModel] = localPrepareModelSpace(data, map_U, ModelOpts);

    ModelOptsOut.Indices = aPC_GenerateTerms(ModelOptsOut);
    [Psi, poly_coeffs, univ_vals] = aPC_PCE_build_Psi(ModelOptsOut, data_model);

    basis_info = struct();
    basis_info.Indices = ModelOptsOut.Indices;
    basis_info.poly_coeffs = poly_coeffs;
    basis_info.univ_vals = univ_vals;

    model_type = lower(strtrim(ModelOptsOut.ModelType));
    if strcmp(model_type, 'apc')
        results = aPC_RegressionMethods(Psi, data_model.Y_train, ModelOptsOut.RegressionMethod);
        results.ModelType = 'apc';
        results.PCKType = 'none';
    else
        if strcmp(model_type, 'geapck') || strcmp(model_type, 'geapck_lowrank')
            % GEAPCK needs response gradients mapped into the same reduced/model
            % space used to build the covariance system.
            [data_model.dY_model, data_model.dYdX_train] = ...
                aPC_PrepareModelGradients(data_grad, map_U, ReduceModel, ModelOptsOut);
        end
        if strcmp(model_type, 'geapck_lowrank')
            results = aPC_PCKBackendLowRankGE(Psi, data_model, ModelOptsOut, basis_info);
        else
            results = aPC_PCKBackend(Psi, data_model, ModelOptsOut, basis_info);
        end
    end

    candidate = struct();
    candidate.results = results;
    candidate.ModelOpts = ModelOptsOut;
    candidate.ReduceModel = ReduceModel;
    candidate.data_model = data_model;
    candidate.poly = poly_coeffs;
    candidate.univ = univ_vals;
    candidate.score = results.LOO;
end

function [data_model, data_grad, ModelOptsOut, ReduceModel] = localPrepareModelSpace(data, map_U, ModelOpts)
    data_grad = data;
    if localNeedsGradientBeforeReduction(ModelOpts)
        [data_grad.dYdU_train, data_grad.dYdX_train] = aPC_PrepareUGradients(data_grad, map_U, ModelOpts);
    end

    if isfield(ModelOpts, 'red_dim') && ~isempty(ModelOpts.red_dim)
        [data_model, ModelOptsOut, ReduceModel] = aPC_dimReduction(data_grad, ModelOpts);
    else
        data_model = data_grad;
        ModelOptsOut = ModelOpts;
        ReduceModel = [];
        if ~isfield(ModelOptsOut, 'num_vars') || isempty(ModelOptsOut.num_vars)
            ModelOptsOut.num_vars = size(data_model.U, 2);
        end
    end
end

function tf = localNeedsGradientBeforeReduction(ModelOpts)
    tf = strcmpi(strtrim(ModelOpts.ModelType), 'geapck') || strcmpi(strtrim(ModelOpts.ModelType), 'geapck_lowrank');
    if tf || ~isfield(ModelOpts, 'ReductionMethod') || isempty(ModelOpts.ReductionMethod)
        return;
    end

    method = lower(strtrim(ModelOpts.ReductionMethod));
    tf = strcmp(method, 'as_pls') || strcmp(method, 'as') || strcmp(method, 'sas');
end
