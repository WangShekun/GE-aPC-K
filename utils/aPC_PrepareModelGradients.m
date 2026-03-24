function [dY_model, dYdX] = aPC_PrepareModelGradients(data, map_U, ReduceModel, ModelOpts)

    % Prepare response gradients for GEAPCK in the exact coordinate system used by
    % the backend: physical X -> standardized U -> optional reduced coordinates.

    if ~isfield(ModelOpts, 'GradientMode') || isempty(ModelOpts.GradientMode)
        ModelOpts.GradientMode = 'auto';
    end

    mode = lower(strtrim(ModelOpts.GradientMode));
    has_grad = isfield(data, 'dYdX_train') && ~isempty(data.dYdX_train);

    switch mode
        case 'provided'
            if ~has_grad
                error('aPC:MissingProvidedGradient', 'GradientMode=provided but data.dYdX_train is missing.');
            end
            dYdX = data.dYdX_train;
        case 'finite-difference'
            dYdX = aPC_FDGradients(data, ModelOpts);
        case 'auto'
            if has_grad
                dYdX = data.dYdX_train;
            else
                dYdX = aPC_FDGradients(data, ModelOpts);
            end
        otherwise
            error('aPC:InvalidGradientMode', 'Unsupported GradientMode: %s', ModelOpts.GradientMode);
    end

    if size(dYdX, 2) ~= size(data.X_train, 2)
        error('aPC:GradientSizeMismatch', 'dYdX_train must match X_train dimension.');
    end

    idx = map_U.nonConstIdx;
    sigma = map_U.sigma;
    % Since U = (X - mu) ./ sigma, gradients transform as dY/dU = dY/dX .* sigma.
    dYdU = dYdX(:, idx) .* sigma;

    if isempty(ReduceModel)
        dY_model = dYdU;
        return;
    end

    [A, ok] = localGetLinearMap(ReduceModel);
    if ~ok
        error('aPC:UnsupportedGEReduction', ['GEAPCK currently supports only reduction methods ' ...
            'with an explicit linear map.']);
    end

    dY_raw_red = dYdU * A;
    if isfield(ReduceModel, 'sigma2') && ~isempty(ReduceModel.sigma2)
        dY_model = dY_raw_red .* ReduceModel.sigma2;
    else
        dY_model = dY_raw_red;
    end
end

function [A, ok] = localGetLinearMap(ReduceModel)
    ok = false;
    A = [];

    if ~isfield(ReduceModel, 'params')
        return;
    end

    p = ReduceModel.params;
    if isfield(p, 'V')
        A = p.V;
        ok = true;
        return;
    end
    if isfield(p, 'W') && isfield(ReduceModel, 'method') && ...
            (strcmpi(ReduceModel.method, 'pls') || ...
             strcmpi(ReduceModel.method, 'spca-primal') || ...
             strcmpi(ReduceModel.method, 'spca-dual'))
        A = p.W;
        ok = true;
        return;
    end
    if isfield(p, 'R')
        A = p.R;
        ok = true;
        return;
    end
    if isfield(p, 'Wwhite') && isfield(p, 'W')
        A = p.Wwhite * p.W;
        ok = true;
        return;
    end
    if isfield(p, 'Pwhite') && isfield(p, 'W')
        A = p.Pwhite * p.W';
        ok = true;
        return;
    end
end
