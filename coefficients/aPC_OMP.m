function results = aPC_OMP(Psi, Y)
% RESULTS = UQ_OMP(PSI, Y)
% Orthogonal Matching Pursuit with fixed default options
%
% All options are hard-coded:
%   precision      = 0
%   early_stop     = 1
%   modified_loo   = 1
%   display        = 0
%   keepiterations = 0
%
% No OPTIONS structure is accepted or used.

%% ===================== Fixed default options =====================
precision       = 0;
early_stop      = 1;
modified_loo    = 1;
display_level   = 0;
keepiterations  = 0;

%% ===================== OBTAIN INFORMATION ABOUT MATRIX DIMENSIONS =====================
[n, P] = size(Psi);

% Ensure Y is a column vector
if size(Y,1) ~= numel(Y)
    Y = Y';
end

% Ensure consistency between Psi and Y
if n ~= numel(Y)
    Psi = Psi';
    [n, P] = size(Psi);
end

%% ===================== INITIALIZATION (Normalization) =====================
% Normalize columns of Psi
normPsi = sqrt(sum(Psi.^2, 1));
PsiNorm = Psi ./ repmat(normPsi, n, 1);

% Initialize residual
R = Y;
normY = sqrt(Y' * Y);
r = Y / normY;

%% ===================== CHECK FOR CONSTANT REGRESSORS =====================
constindices = find(~any(diff(Psi,1)));
bool_const   = ~isempty(constindices);

%% ===================== OMP MAIN LOOP =====================
ind     = [];
indtot = 1:P;
M       = [];
kmax    = min(n, P);

LOO         = Inf(kmax,1);
normEmpErr  = Inf(kmax,1);
coeff       = zeros(P, kmax);
indit       = zeros(kmax, kmax);

count       = 1;
k           = 0.1;
cond_early  = 1;
LOOmin      = Inf;
refscore    = Inf;

while (norm(R) > precision) && (count <= kmax) ...
        && xor((cond_early | early_stop), ~cond_early)

    % Remove previously selected index
    if count ~= 1
        indtot(iindx) = [];
    end

    % Correlation with residual
    h = abs(r' * PsiNorm);
    [~, iindx] = max(h(indtot));
    indx = indtot(iindx);

    % Force constant regressor at first iteration if present
    if (count == 1) && bool_const
        iindx = constindices(1);
        indx  = indtot(iindx);
    end

    % Update information matrix
    if count == 1
        M = 1 / (Psi(:,indx)' * Psi(:,indx));
    else
        x = Psi(:,ind)' * Psi(:,indx);
        rloc = Psi(:,indx)' * Psi(:,indx);
        M = aPC_blockwise_inverse(M, x, x', rloc);
    end

    % Update active set
    ind = [ind; indx];
    indit(1:count, count) = ind;

    % Projection
    Xpro = Psi(:, ind);
    TT   = Xpro' * Y;
    beta = M * TT;

    coeff(ind, count) = beta;

    % LOO error
    [Loo_out, normEmpErr_out, temp] = ...
        aPC_PCE_loo_error(Xpro, M, Y, beta, modified_loo);

    LOO(count)        = Loo_out;
    normEmpErr(count) = normEmpErr_out;

    if Loo_out < refscore
        refscore = Loo_out;
        optErrorParams = temp;
    end

    % Update residual
    R = Y - Xpro * beta;
    r = R / sqrt(R' * R);

    % Early stopping logic
    countinf  = max(1, floor(count - k * kmax));
    LOOmin    = min(LOOmin, LOO(count));
    cond_early = (min(LOO(countinf:count)) <= LOOmin);

    count = count + 1;

    % Debug display
    if display_level > 3
        fprintf('OMP Iteration Nr. %d\n', count-1);
    end
end

%% ===================== POST-PROCESSING =====================
LOO = LOO(1:count-1);
countmin = find(LOO == min(LOO), 1);

coefficients     = coeff(:, countmin);
LOOmin           = LOO(countmin);
normEmpErrmin    = normEmpErr(countmin);

%% ===================== OUTPUT STRUCTURE =====================
results.coefficients     = coefficients;
results.best_basis_index = countmin;
results.max_score        = 1 - LOOmin;
results.LOO              = LOOmin;
results.normEmpErr       = normEmpErrmin;
results.nz_idx           = indit(1:countmin, countmin)';

if ~keepiterations
    results.a_scores   = 1 - LOO(1:countmin)';
    results.coeff_array = coeff(:, 1:countmin);
else
    results.a_scores   = 1 - LOO';
    results.coeff_array = coeff;
end

results.optErrorParams = optErrorParams;
results.RegressionMethod = 'OMP';
end

function M = aPC_blockwise_inverse(Ainv,B,C,D)
% Minv = UQ_BLOCKWISE_INVERSE(AINV,B,C,D): perform blockwise inversion of the
%     non-singular square matrix M defined as M = [[A B]; [C D]] . AINV is the
%     inverse of the square-submatrix A. B, C and D can have any dimension,
%     provided their combination defines a square matrix M.


%% INPUT ARGUMENT CHECKS
% all inputs must be defined
if(nargin~=4)
    error('%s: all 4 input arguments must be given.', mfilename);
end

% Ainv must be square
if(size(Ainv,1) ~= size(Ainv,2))
    error('%s: Input argument Ainv must be a square matrix.', mfilename);
end

%% Calculate useful blocks:
% Schur complement SC = (D - C*Ainv*B). Same dimension
% of D
if isscalar(D)
    % Inverse of D
    Dinv = 1/D;
    % Schur complement
    SCinv = 1/(D - C*Ainv*B);
else
    % Inverse of D
    Dinv = D\eye(size(D));
    % Schur complement
    SCinv = (D - C*Ainv*B)\eye(size(D));
end
% Caching a couple of products to improve efficiency
T1 = Ainv*B*SCinv;
T2 = C*Ainv;

% Assembly the inverse
M = [Ainv+T1*T2 -T1;
    -(SCinv)*T2     SCinv];
end
