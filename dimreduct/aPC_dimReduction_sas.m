function [X_red, ReduceModel] = aPC_dimReduction_sas(U, Y, dYdU, ncomp, ModelOpts)
% Supervised Active Subspace (SAS) reduction using response-weighted
% gradient covariance.
%
% Compared with classical AS, SAS introduces response-based sample weights
% so that gradients associated with more informative response samples have
% larger influence on the extracted subspace.
%
% Inputs
%   U        : [N x M] input samples
%   Y        : [N x 1] scalar responses
%   dYdU     : [N x M] gradients in U-space
%   ncomp    : reduced dimension
%   ModelOpts: optional struct; method options are read from
%              ModelOpts.sas_opts
%
%   sas_opts : optional struct with fields
%       .weight_method  - 'zpower' (default) | 'density'
%       -- zpower method --
%       .response_power - exponent used in response weighting
%                         default: 2
%       .weight_eps     - small positive regularization for weights
%                         default: 1e-12
%       -- density method --
%       .gamma          - sparse-region emphasis strength (>=0)
%                         default: 1.0
%       .sigma_y        - response-kernel bandwidth
%                         default: std(Y)
%
% Weight definitions
%   zpower:
%   z_i = (y_i - mean(Y)) / std(Y)
%   w_i = abs(z_i).^response_power + weight_eps
%
%   density:
%   rho_i = (1/N) * sum_j exp(-(y_i - y_j)^2 / (2*sigma_y^2))
%   w_i   = 1 + gamma * (1 - normalize01(rho_i))
%
% Outputs
%   X_red        : [N x ncomp] reduced coordinates
%   ReduceModel  : struct with projection and transform handles
%
% Notes
%   - If Y has nearly zero variance, SAS falls back to classical AS.
%   - For downstream GE surrogate modeling:
%         dYdU_red = dYdU * A

    opts = struct();
    if nargin >= 5 && ~isempty(ModelOpts) ...
            && isfield(ModelOpts, 'sas_opts') ...
            && ~isempty(ModelOpts.sas_opts)
        opts = ModelOpts.sas_opts;
    end

    [N, M] = size(U);

    if isempty(Y)
        error('aPC:ReductionResponseRequired', ...
            'sas requires response Y.');
    end
    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', ...
            'sas requires gradients in U-space.');
    end

    Y = Y(:);

    if size(Y, 1) ~= N
        error('aPC:InvalidYSize', ...
            'Y must have the same number of rows as U.');
    end
    if size(dYdU, 1) ~= N || size(dYdU, 2) ~= M
        error('aPC:InvalidGradientSize', ...
            'dYdU must have size [size(U,1), size(U,2)].');
    end
    if numel(ncomp) ~= 1 || ncomp <= 0 || floor(ncomp) ~= ncomp || ncomp > M
        error('aPC:InvalidSASDim', ...
            'sas red_dim must be an integer in [1, size(U,2)].');
    end

    weight_method = lower(strtrim(char(getOption(opts, 'weight_method', 'zpower'))));
    response_power = [];
    weight_eps = [];
    gamma = [];
    sigma_y = [];
    rho = [];
    rho_tilde = [];

    if ~strcmp(weight_method, 'zpower') && ~strcmp(weight_method, 'density')
        error('aPC:InvalidSASWeightMethod', ...
            'sas weight_method must be ''zpower'' or ''density''.');
    end

    % ---------------------------------------------------------------------
    % Response-based weights
    % ---------------------------------------------------------------------
    y_mean = mean(Y);
    y_std = std(Y, 0, 1);

    if y_std < eps
        % Fallback to classical AS if response has nearly zero variance
        weights = ones(N, 1);
    else
        if strcmp(weight_method, 'zpower')
            response_power = getOption(opts, 'response_power', 2);
            weight_eps = getOption(opts, 'weight_eps', 1e-12);

            if ~isscalar(response_power) || response_power <= 0
                error('aPC:InvalidSASPower', ...
                    'sas response_power must be a positive scalar.');
            end
            if ~isscalar(weight_eps) || weight_eps < 0
                error('aPC:InvalidSASEps', ...
                    'sas weight_eps must be a nonnegative scalar.');
            end

            z = (Y - y_mean) / y_std;
            weights = abs(z).^response_power + weight_eps;
        else
            gamma = getOption(opts, 'gamma', 2);
            sigma_y = getOption(opts, 'sigma_y', y_std);

            if ~isscalar(gamma) || gamma < 0
                error('aPC:InvalidSASGamma', ...
                    'sas gamma must be a nonnegative scalar.');
            end
            if ~isscalar(sigma_y) || sigma_y <= 0
                error('aPC:InvalidSASSigma', ...
                    'sas sigma_y must be a positive scalar.');
            end

            dy = Y - Y';
            rho = mean(exp(-(dy.^2) / (2 * sigma_y^2)), 2);
            rho_tilde = normalize01(rho);
            weights = 1 + gamma * (1 - rho_tilde);
        end
    end

    wsum = sum(weights);
    if wsum <= 0
        weights = ones(N, 1);
        wsum = N;
    end

    % ---------------------------------------------------------------------
    % Weighted gradient covariance
    %   C = sum_i w_i * g_i' * g_i / sum_i w_i
    % ---------------------------------------------------------------------
    Gw = dYdU .* weights;              % each row scaled by sample weight
    C = (dYdU' * Gw) / wsum;
    C = 0.5 * (C + C');

    [V, D] = eig(C);
    [eigvals, idx] = sort(diag(D), 'descend');
    A = V(:, idx(1:ncomp));

    X_red = U * A;

    params = struct();
    params.A = A;
    params.eigvals = eigvals;
    params.weights = weights;
    params.weight_method = weight_method;
    params.response_power = response_power;
    params.weight_eps = weight_eps;
    params.gamma = gamma;
    params.sigma_y = sigma_y;
    params.rho = rho;
    params.rho_tilde = rho_tilde;
    params.y_mean = y_mean;
    params.y_std = y_std;

    ReduceModel = struct();
    ReduceModel.method = 'sas';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) U_new * params.A;
    ReduceModel.transformGradient = @(dYdU_new) dYdU_new * params.A;
end


function val = getOption(opts, name, defaultVal)
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = defaultVal;
    end
end

function x01 = normalize01(x)
    xmin = min(x);
    xmax = max(x);
    if xmax - xmin <= eps
        x01 = ones(size(x));
    else
        x01 = (x - xmin) / (xmax - xmin);
    end
end
