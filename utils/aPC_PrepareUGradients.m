function [dYdU, dYdX] = aPC_PrepareUGradients(data, map_U, ModelOpts)

    % Prepare response gradients in standardized U-space before any optional
    % dimension reduction. This helper is shared by GEAPCK itself and any
    % reduction method that needs gradient information.

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
end
