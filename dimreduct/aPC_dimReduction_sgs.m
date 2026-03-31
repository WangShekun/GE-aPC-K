function [X_red, ReduceModel] = aPC_dimReduction_sgs(U, Y, dYdU, ncomp, ModelOpts)
% Joint dimension reduction using both response Y and gradients dYdU.
%
% This method builds a fused covariance-like matrix:
%   C_joint = alpha * Cg_norm + (1-alpha) * Cy_norm
% where
%   Cg : gradient covariance (active-subspace style)
%   Cy : response-driven local variation matrix
%
% Inputs
%   U      : [N x M] input samples
%   Y      : [N x 1] responses
%   dYdU   : [N x M] gradients in U-space
%   ncomp  : reduced dimension
%   ModelOpts : optional struct; options are read from ModelOpts.sgs_opts
%   sgs_opts : optional struct with fields
%       .alpha      - weight for gradient part in [0,1], default 0.5
%       .bandwidth  - Gaussian bandwidth for local weighting; if empty,
%                     uses median pairwise distance
%       .eps_reg    - small regularization for denominator, default 1e-12
%       .normalize  - true/false, trace-normalize Cg and Cy, default true
%       .centerU    - true/false, center U before pairwise distances, default false
%
% Outputs
%   X_red        : [N x ncomp] reduced coordinates
%   ReduceModel  : struct with projection and transform handle
%
% Notes
%   - This is a linear projection method.
%   - It explicitly uses both Y and dYdU.
%   - Cy is inspired by local response variation, so it can capture
%     nonlinear response trends better than a purely global covariance.

    opts = struct();
    if nargin >= 5 && ~isempty(ModelOpts) && isfield(ModelOpts, 'sgs_opts') && ~isempty(ModelOpts.sgs_opts)
        opts = ModelOpts.sgs_opts;
    end

    [N, M] = size(U);

    if isempty(Y)
        error('aPC:ReductionResponseRequired', ...
            'joint reduction requires response Y.');
    end
    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', ...
            'joint reduction requires gradients in U-space.');
    end
    if size(Y, 1) ~= N
        error('aPC:InvalidYSize', ...
            'Y must have the same number of rows as U.');
    end
    if size(dYdU, 1) ~= N || size(dYdU, 2) ~= M
        error('aPC:InvalidGradientSize', ...
            'dYdU must have size [size(U,1), size(U,2)].');
    end
    if numel(ncomp) ~= 1 || ncomp <= 0 || floor(ncomp) ~= ncomp || ncomp > M
        error('aPC:InvalidJointDim', ...
            'joint red_dim must be an integer in [1, size(U,2)].');
    end

    Y = Y(:);

    alpha = getOption(opts, 'alpha', 0.5);
    if ~isscalar(alpha) || alpha < 0 || alpha > 1
        error('aPC:InvalidAlpha', 'opts.alpha must be in [0,1].');
    end

    eps_reg = getOption(opts, 'eps_reg', 1e-12);
    normalize = getOption(opts, 'normalize', true);
    centerU = getOption(opts, 'centerU', false);
    bandwidth = getOption(opts, 'bandwidth', []);

    if centerU
        U_work = U - mean(U, 1);
    else
        U_work = U;
    end

    % ---------------------------------------------------------------------
    % Part 1: gradient covariance (Active Subspace style)
    % ---------------------------------------------------------------------
    Cg = (dYdU' * dYdU) / max(N, 1);
    Cg = 0.5 * (Cg + Cg');

    % ---------------------------------------------------------------------
    % Part 2: response-driven local variation matrix
    %
    % Cy = sum_ij w_ij * s_ij^2 * du_ij * du_ij'
    % where
    %   s_ij = (Y_j - Y_i) / (||du_ij||^2 + eps)
    %
    % This emphasizes directions along which local changes in input
    % induce strong changes in output.
    % ---------------------------------------------------------------------
    if N == 1
        Cy = zeros(M, M);
        Wsum = 1.0;
    else
        % Pairwise squared distances
        UU = sum(U_work.^2, 2);
        D2 = max(UU + UU' - 2 * (U_work * U_work'), 0);

        if isempty(bandwidth)
            dvec = sqrt(D2(triu(true(N), 1)));
            dvec = dvec(isfinite(dvec) & dvec > 0);
            if isempty(dvec)
                bandwidth = 1.0;
            else
                bandwidth = median(dvec);
            end
        end

        if ~isscalar(bandwidth) || bandwidth <= 0
            error('aPC:InvalidBandwidth', ...
                'opts.bandwidth must be a positive scalar.');
        end

        W = exp(-D2 / (2 * bandwidth^2));
        W(1:N+1:end) = 0;  % remove diagonal
        Wsum = sum(W(:));
        if Wsum <= 0
            Wsum = 1.0;
        end

        Cy = zeros(M, M);

        % O(N^2 M), consistent with small/medium-sample surrogate settings.
        for i = 1:N
            du = U_work - U_work(i, :);              % [N x M]
            dy = Y - Y(i);                           % [N x 1]
            denom = sum(du.^2, 2) + eps_reg;         % [N x 1]
            coeff = W(:, i) .* (dy.^2 ./ (denom.^2));% [N x 1]

            % Accumulate du' * diag(coeff) * du without forming diag()
            Cy = Cy + du' * (du .* coeff);
        end

        Cy = Cy / Wsum;
        Cy = 0.5 * (Cy + Cy');
    end

    % ---------------------------------------------------------------------
    % Optional normalization to balance the two sources
    % ---------------------------------------------------------------------
    if normalize
        trCg = trace(Cg);
        trCy = trace(Cy);

        if trCg > eps_reg
            Cg = Cg / trCg;
        end
        if trCy > eps_reg
            Cy = Cy / trCy;
        end
    end

    % ---------------------------------------------------------------------
    % Joint matrix and eigendecomposition
    % ---------------------------------------------------------------------
    Cjoint = alpha * Cg + (1 - alpha) * Cy;
    Cjoint = 0.5 * (Cjoint + Cjoint');

    [V, D] = eig(Cjoint);
    [eigvals, idx] = sort(diag(D), 'descend');
    A = V(:, idx(1:ncomp));

    X_red = U * A;

    % ---------------------------------------------------------------------
    % Store model
    % ---------------------------------------------------------------------
    params = struct();
    params.A = A;
    params.alpha = alpha;
    params.bandwidth = bandwidth;
    params.eps_reg = eps_reg;
    params.normalize = normalize;
    params.centerU = centerU;
    params.Cg = Cg;
    params.Cy = Cy;
    params.Cjoint = Cjoint;
    params.eigvals = eigvals;

    ReduceModel = struct();
    ReduceModel.method = 'sgs';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) U_new * params.A;
end

function val = getOption(opts, name, defaultVal)
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = defaultVal;
    end
end
