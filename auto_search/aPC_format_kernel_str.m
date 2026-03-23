function kernel_str = aPC_format_kernel_str(ModelOpts)
% aPC_format_kernel_str
% -------------------------------------------------------
% Format kernel hyperparameters for display.
% Supports both KPCA (X-kernel only) and SKPCA (X- and Y-kernel).
% Robust to cell/char/empty inputs. Designed for auto-search printing.
% -------------------------------------------------------

    % -------- helper: cell → char --------
    function v = read_field(name)
        if ~isfield(ModelOpts,name) || isempty(ModelOpts.(name))
            v = [];
            return;
        end
        val = ModelOpts.(name);
        if iscell(val), v = val{1};
        else,           v = val;
        end
    end

    % =====================================================================
    %                          X-KERNEL (input kernel)
    % =====================================================================
    ktype = read_field('kernel_type');

    if isempty(ktype)
        kx_str = 'kernel=-';
    else
        gammaX  = read_field('kernel_gamma');
        degreeX = read_field('kernel_degree');
        offsetX = read_field('kernel_offset');

        params = {};
        if ~isempty(gammaX)
            params{end+1} = sprintf('γ=%g', gammaX);
        end
        if ~isempty(degreeX)
            params{end+1} = sprintf('d=%g', degreeX);
        end
        if ~isempty(offsetX)
            params{end+1} = sprintf('c=%g', offsetX);
        end

        if isempty(params)
            kx_str = sprintf('kernel=%s', ktype);
        else
            kx_str = sprintf('kernel=%s(%s)', ktype, strjoin(params, ', '));
        end
    end


    % =====================================================================
    %                       Y-KERNEL (label kernel, only for SKPCA)
    % =====================================================================
    ltype = read_field('label_kernel_type');

    if isempty(ltype)
        % KPCA 情况
        ky_str = 'y-kernel=-';
    else
        gammaY  = read_field('label_kernel_gamma');
        degreeY = read_field('label_kernel_degree');
        offsetY = read_field('label_kernel_offset');

        paramsY = {};
        if ~isempty(gammaY)
            paramsY{end+1} = sprintf('γ=%g', gammaY);
        end
        if ~isempty(degreeY)
            paramsY{end+1} = sprintf('d=%g', degreeY);
        end
        if ~isempty(offsetY)
            paramsY{end+1} = sprintf('c=%g', offsetY);
        end

        if isempty(paramsY)
            ky_str = sprintf('y-kernel=%s', ltype);
        else
            ky_str = sprintf('y-kernel=%s(%s)', ltype, strjoin(paramsY, ', '));
        end
    end

    % =====================================================================
    %                         COMBINE (KPCA / SKPCA)
    % =====================================================================
    kernel_str = sprintf('%s | %s', kx_str, ky_str);

end
