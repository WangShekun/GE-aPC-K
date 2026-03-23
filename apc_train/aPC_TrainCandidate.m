function candidate = aPC_TrainCandidate(data, map_U, ModelOpts)
% Train one candidate configuration in the current model space.

    [data_model, ModelOptsOut, ReduceModel] = localPrepareModelSpace(data, ModelOpts);

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
        if strcmp(model_type, 'geapck')
            [data_model.dY_model, data_model.dYdX_train] = ...
                aPC_PrepareModelGradients(data, map_U, ReduceModel, ModelOptsOut);
        end
        results = aPC_PCKBackend(Psi, data_model.Y_train, data_model, ModelOptsOut, basis_info);
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

function [data_model, ModelOptsOut, ReduceModel] = localPrepareModelSpace(data, ModelOpts)
    if isfield(ModelOpts, 'red_dim') && ~isempty(ModelOpts.red_dim)
        [data_model, ModelOptsOut, ReduceModel] = aPC_dimReduction(data, ModelOpts);
    else
        data_model = data;
        ModelOptsOut = ModelOpts;
        ReduceModel = [];
        if ~isfield(ModelOptsOut, 'num_vars') || isempty(ModelOptsOut.num_vars)
            ModelOptsOut.num_vars = size(data_model.U, 2);
        end
    end
end
