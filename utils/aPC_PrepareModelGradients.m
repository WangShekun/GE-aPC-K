function [dY_model, dYdX] = aPC_PrepareModelGradients(data, map_U, ReduceModel, ModelOpts)

    % Prepare response gradients for GEAPCK in the exact coordinate system used by
    % the backend: physical X -> standardized U -> optional reduced coordinates.

    if isfield(data, 'dYdU_train') && ~isempty(data.dYdU_train)
        dYdU = data.dYdU_train;
        if isfield(data, 'dYdX_train') && ~isempty(data.dYdX_train)
            dYdX = data.dYdX_train;
        else
            dYdX = [];
        end
    else
        [dYdU, dYdX] = aPC_PrepareUGradients(data, map_U, ModelOpts);
    end

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

    if ~isfield(ReduceModel, 'method') || isempty(ReduceModel.method) || ...
            ~isfield(ReduceModel, 'params')
        return;
    end

    method = lower(strtrim(ReduceModel.method));
    p = ReduceModel.params;
    if strcmp(method, 'pls')
        if isfield(p, 'W')
            A = p.W;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'pca')
        if isfield(p, 'V')
            A = p.V;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'spca_hsic')
        if isfield(p, 'W')
            A = p.W;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'rp')
        if isfield(p, 'R')
            A = p.R;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'mcp')
        if isfield(p, 'Wwhite') && isfield(p, 'W')
            A = p.Wwhite * p.W;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'ica')
        if isfield(p, 'Pwhite') && isfield(p, 'W')
            A = p.Pwhite * p.W';
            ok = true;
        end
        return;
    end

    if strcmp(method, 'gepls')
        if isfield(p, 'A')
            A = p.A;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'as_pls')
        if isfield(p, 'A')
            A = p.A;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'as')
        if isfield(p, 'A')
            A = p.A;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'sas')
        if isfield(p, 'A')
            A = p.A;
            ok = true;
        end
        return;
    end

    if strcmp(method, 'sgs')
        if isfield(p, 'A')
            A = p.A;
            ok = true;
        end
        return;
    end
end
