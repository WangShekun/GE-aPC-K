function dYdX = aPC_FDGradients(data, ModelOpts)

    if ~isfield(data, 'ModelFcn') || isempty(data.ModelFcn)
        error('aPC:MissingModelFcn', 'Automatic gradients require data.ModelFcn.');
    end

    if ~isfield(ModelOpts, 'FDOpts') || isempty(ModelOpts.FDOpts)
        FDOpts = struct();
    else
        FDOpts = ModelOpts.FDOpts;
    end

    FDOpts = localSetDef(FDOpts, 'type', 'forward');
    FDOpts = localSetDef(FDOpts, 'step', 1e-6);

    X = data.X_train;
    Y = data.Y_train(:);
    [N, d] = size(X);

    dYdX = zeros(N, d);
    for j = 1:d
        h = FDOpts.step * max(1, std(X(:, j)));
        Xp = X;
        Xp(:, j) = Xp(:, j) + h;
        Yp = data.ModelFcn(Xp);
        Yp = Yp(:);

        if strcmpi(FDOpts.type, 'central')
            Xm = X;
            Xm(:, j) = Xm(:, j) - h;
            Ym = data.ModelFcn(Xm);
            Ym = Ym(:);
            dYdX(:, j) = (Yp - Ym) / (2 * h);
        else
            dYdX(:, j) = (Yp - Y) / h;
        end
    end
end

function S = localSetDef(S, field, val)
    if ~isfield(S, field) || isempty(S.(field))
        S.(field) = val;
    end
end
