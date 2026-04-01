function [X_red, ReduceModel] = aPC_dimReduction_gepls(U, Y, dYdU, ncomp, ModelOpts)
% Gradient-Enhanced Partial Least Squares (GEPLS) dimension reduction.
%
% GEPLS is a single-stage supervised linear dimension reduction method
% that jointly exploits response values and gradient observations through
% a multi-response PLS framework.
%
% Inputs
%   U      : [N x M] input samples in U-space
%   Y      : [N x 1] scalar responses
%   dYdU   : [N x M] gradients in U-space
%   ncomp  : reduced dimension
%   ModelOpts : optional struct; method options are read from
%               ModelOpts.gepls_opts
%
%   gepls_opts : optional struct with fields
%       .gamma - gradient block weight in augmented response
%                default: 1.0
%
% Outputs
%   X_red        : [N x ncomp] reduced coordinates
%   ReduceModel  : struct with projection and transform handles
%
% Notes
%   - This implementation requires MATLAB plsregress.
%   - The final projection basis A is orthonormal in the original U-space.
%   - For downstream GE surrogate modeling:
%         dYdU_red = dYdU * A

    opts = struct();
    if nargin >= 5 && ~isempty(ModelOpts) ...
            && isfield(ModelOpts, 'gepls_opts') ...
            && ~isempty(ModelOpts.gepls_opts)
        opts = ModelOpts.gepls_opts;
    end

    [N, M] = size(U);

    if isempty(Y)
        error('aPC:ReductionResponseRequired', ...
            'gepls requires response Y.');
    end
    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', ...
            'gepls requires gradients in U-space.');
    end
    if exist('plsregress', 'file') ~= 2
        error('aPC:PLSRegressUnavailable', ...
            'gepls requires MATLAB plsregress (Statistics and Machine Learning Toolbox).');
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
        error('aPC:InvalidGEPLSDim', ...
            'gepls red_dim must be an integer in [1, size(U,2)].');
    end
    if N < 2
        error('aPC:TooFewSamples', ...
            'gepls requires at least two samples.');
    end
    if ncomp > N - 1
        error('aPC:InvalidGEPLSDim', ...
            'gepls red_dim must satisfy red_dim <= size(U,1)-1.');
    end

    gamma = getOption(opts, 'gamma', 0.3);
    if ~isscalar(gamma) || gamma < 0
        error('aPC:InvalidGEPLSGamma', ...
            'opts.gamma must be a nonnegative scalar.');
    end

    % ---------------------------------------------------------------------
    % 1) Center and whiten input
    % ---------------------------------------------------------------------
    muU = mean(U, 1);
    Uc = bsxfun(@minus, U, muU);

    whiten_reg = 1e-10;
    Su = (Uc' * Uc) / max(N - 1, 1);
    Su = 0.5 * (Su + Su') + whiten_reg * eye(M);

    [L, pchol] = chol(Su, 'lower');
    if pchol ~= 0
        [V, D] = eig(Su);
        d = max(diag(D), whiten_reg);
        L = V * diag(sqrt(d)) * V';
    end

    X = Uc / L;

    % Gradient mapping to whitened space:
    % X = Uc / L  =>  dY/dX = dY/dU * L
    Gx = dYdU * L;

    % ---------------------------------------------------------------------
    % 2) Build augmented response matrix
    %    R = [Ys, gamma * Gs / sqrt(M)]
    % ---------------------------------------------------------------------
    Ys = safeStandardize(Y);
    Gs = safeStandardize(Gx);

    R = [Ys, (gamma / sqrt(max(M, 1))) * Gs];

    % ---------------------------------------------------------------------
    % 3) Multi-response PLS via MATLAB plsregress
    % ---------------------------------------------------------------------
    [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, R, ncomp);

    W = stats.W(:, 1:ncomp);
    P = XL(:, 1:ncomp);
    T = XS(:, 1:ncomp);
    Q = YL(:, 1:ncomp);

    Rmat = P' * W;
    if rcond(Rmat) < 1e-12
        Wstar = W * pinv(Rmat);
    else
        Wstar = W / Rmat;
    end

    % ---------------------------------------------------------------------
    % 4) Map basis back to original U-space and orthonormalize
    % ---------------------------------------------------------------------
    Araw = L \ Wstar(:, 1:ncomp);
    [A, ~] = qr(Araw, 0);

    X_red = Uc * A;

    % ---------------------------------------------------------------------
    % Store model
    % ---------------------------------------------------------------------
    params = struct();
    params.A = A;
    params.Araw = Araw;
    params.muU = muU;
    params.L = L;
    params.gamma = gamma;
    params.ncomp = ncomp;
    params.W = W;
    params.P = P;
    params.T = T;
    params.Q = Q;
    params.Wstar = Wstar;
    params.XL = XL;
    params.YL = YL;
    params.XS = XS;
    params.YS = YS;
    params.BETA = BETA;
    params.PCTVAR = PCTVAR;
    params.MSE = MSE;
    params.stats = stats;

    ReduceModel = struct();
    ReduceModel.method = 'gepls';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) bsxfun(@minus, U_new, params.muU) * params.A;
    ReduceModel.transformGradient = @(dYdU_new) dYdU_new * params.A;
end


function Xs = safeStandardize(X)
    Xc = bsxfun(@minus, X, mean(X, 1));
    s = std(Xc, 0, 1);
    s(s < eps) = 1.0;
    Xs = bsxfun(@rdivide, Xc, s);
end


function val = getOption(opts, name, defaultVal)
    if isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = defaultVal;
    end
end