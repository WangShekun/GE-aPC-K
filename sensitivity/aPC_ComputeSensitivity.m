function SA = aPC_ComputeSensitivity(results, ModelOpts)
% aPC_ComputeSensitivity
% --------------------------------------------------------
% 基于正交归一 aPC-PCE 计算 Sobol 灵敏度（1 阶、2 阶、总效应）
%
% 输入：
%   results.coefficients : K×1 PCE 系数 c_k
%   results.Indices      : K×M 多指数 alpha_k
%   ModelOpts.num_vars   : 输入维数 M
%
% 输出：
%   SA.S_first  : 1×M 一阶 Sobol 指数
%   SA.S_total  : 1×M 总效应 Sobol 指数
%   SA.S_second : M×M 二阶 Sobol 指数矩阵（上三角有值）
% --------------------------------------------------------

    if isfield(results, 'ModelType') && ~strcmpi(results.ModelType, 'apc')
        error('aPC:SensitivityOnlyForAPC', ...
            'aPC_ComputeSensitivity currently supports APC only. Current ModelType = %s.', ...
            results.ModelType);
    end

    c       = results.coefficients(:);
    Indices = results.Indices;
    M       = ModelOpts.num_vars;

    % ---- 找常数项，获取总方差 ----
    isConst = all(Indices == 0, 2);
    if ~any(isConst)
        error('No constant term found in Indices.');
    end
    idx0    = find(isConst, 1, 'first');

    maskVar = true(size(c));
    maskVar(idx0) = false;
    c_eff   = c(maskVar);

    VarTot = sum(c_eff.^2);
    if VarTot <= 0
        warning('Total variance <= 0, Sobol indices may be meaningless.');
        VarTot = max(VarTot, eps);
    end

    % 预分配
    S_first  = zeros(1, M);
    S_total  = zeros(1, M);
    S_second = zeros(M, M);

    % ---- 一阶 & 总效应 ----
    for i = 1:M
        % 一阶：仅依赖 X_i，其他维度指数为 0
        mask_Si = (Indices(:, i) > 0);
        for j = 1:M
            if j ~= i
                mask_Si = mask_Si & (Indices(:, j) == 0);
            end
        end
        mask_Si(idx0) = false;  % 排除常数项

        S_first(i) = sum(c(mask_Si).^2) / VarTot;

        % 总效应：所有含 X_i 的项
        mask_Ti = (Indices(:, i) > 0);
        mask_Ti(idx0) = false;

        S_total(i) = sum(c(mask_Ti).^2) / VarTot;
    end

    % ---- 二阶：只含 X_i, X_j 的交互项 ----
    for i = 1:M
        for j = i+1:M
            mask_Sij = (Indices(:, i) > 0) & (Indices(:, j) > 0);
            for k = 1:M
                if k ~= i && k ~= j
                    mask_Sij = mask_Sij & (Indices(:, k) == 0);
                end
            end
            mask_Sij(idx0) = false;

            S_second(i, j) = sum(c(mask_Sij).^2) / VarTot;
            S_second(j, i) = S_second(i, j);  % 对称
        end
    end

    SA.S_first  = S_first;
    SA.S_total  = S_total;
    SA.S_second = S_second;
    SA.VarTot   = VarTot;
end
