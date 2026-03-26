function results = aPC_LARS(Psi, Y)
% RESULTS = UQ_LAR(PSI, Y): sparse regression with Hybrid Least Angle
%   Regression. PSI is the matrix of evaluations of the regressors in the
%   experimental design points, while Y are the corresponding model
%   responses.
%
%   This version hardcodes the following options:
%     - 'normalize_columns' = 1
%     - 'early_stop' = 1
%     - 'hybrid_lars' = 1
%     - 'modified_loo' = 1
%     - 'hybrid_loo' = 1
%     - 'no_selection' = 0
%     - 'generalized_ls' = 0
%     - 'DisplayLevel' = 0
%
%   The function does not require any input for options now.

%% Hardcoded options
normalize_columns = 1;
early_stop = 1;
hybrid_lars = 1;
modified_loo = 1;
hybrid_loo = 1;
no_selection = 0;
generalized_ls = 0;
DisplayLevel = 0;
use_lars_order = 1;

%% OLS options
olsoptions.modified_loo = modified_loo;

% Get the number of data points and regressors
[N ,P] = size(Psi);

%% Trivial case: only 1 regressor, use OLS solution directly
if P == 1
    % OLS solution
    if generalized_ls
        olsoptions.CY = CY;
        ols_results = aPC_OLS(Psi, Y);
    else
        ols_results = aPC_OLS(Psi, Y);
    end
    
    % Assign the outputs
    results.coefficients = ols_results.coefficients;
    results.LOO = ols_results.LOO;
    results.normEmpErr = ols_results.normEmpErr;
    results.optErrorParams = ols_results.optErrorParams;
    results.coeff_array = results.coefficients;
    results.max_score   = 1-results.LOO;
    results.a_scores    = results.max_score;
    results.loo_scores  = results.LOO;
    results.best_basis_index = 1;
    results.nz_idx  = 1;
    results.LOO_lars = results.LOO;
    results.lars_idx = 1;
    
    % Exit from the algorithm
    return;
end

%% Apply the weight matrix if generalized least squares
if generalized_ls
    % Decorrelate the outputs
    CYinv = CY\eye(size(CY));
    L = chol(CYinv);
    Psi = L * Psi;
    Y = L * Y;
end

%% Centering and normalization

% Check for constant regressors (this is necessary before centering)
constidx = ~any(diff(Psi, 1));
constindices = find(constidx); % Indices of constant regressors
constval = Psi(1,constidx);
bool_const_and_center = any(constidx); % True if constant regressor exists

% Center the data if possible.
if bool_const_and_center
    % Center Psi. This will remove the constant regressors, as they 
    % are treated separately in LARS
    mu_Psi = mean(Psi, 1);
    Psi = Psi - repmat(mu_Psi, N, 1);
    
    % Center the data as well
    mu_Y = mean(Y);
    Y = Y - mu_Y;
    modi_diag = 1 / N * ones(N, 1);
    run_lars_iterations = var(Y); % Used as a flag
else
    modi_diag = zeros(N, 1);
    run_lars_iterations = 1; % Always run LARS iterations if no constant regressor
end

% Normalize the regressors and the experimental design if necessary
if normalize_columns
    normPsi = sqrt(sum(Psi.^2, 1) / (N - 1)); % Columns should have stddev 1 (if centered)
    nznorm_indices = (normPsi ~= 0); % Indices with nonzero norm
    Psi(:, nznorm_indices) = Psi(:, nznorm_indices) ./ repmat(normPsi(nznorm_indices), N, 1);
end

%% Initialization of the LAR iterations

nvars = min(N - 2, P); % Maximum number of active predictors: either the full set of basis elements, or N-1 (the - 2 is due to how k is incremented in the loop)
mu = zeros(size(Y)); % Initial direction of the LAR 
a_coeff = []; % Set of active coefficients
i_coeff = 1:P; % Maximal set of coefficients
M = []; % Initial information matrix (PsiT_j Psi_j)
maxk = 8 * nvars; % Maximum number of LAR iterations
coeff_array = zeros(nvars + 1, P);

a_scores = -inf(1, nvars + 1);
loo_scores = inf * ones(1, nvars + 1);

% Initialize the best LARS leave-one-out score
refscore = inf;

%% Iterative LAR
k = 0;
if DisplayLevel > 1
    fprintf('Maximum LARS candidate basis size: %d\n', P);
end

% Initial score: just do OLS with the constant term, if it exists
if bool_const_and_center
    ols_results = aPC_OLS(ones(size(Y, 1), 1), Y);
    loo_scores(1) = ols_results.LOO;
    a_scores(1) = 1 - loo_scores(1);
end

maxiter = min(maxk, nvars);

while k < maxiter && run_lars_iterations
    k = k + 1;
    if DisplayLevel > 3
        fprintf('Computing LAR iteration %d\n', k);
    end
    
    % Correlation with the residual
    cj = Psi' * (Y - mu);
    % Getting the most correlated with the current inactive set
    [C, idx] = max(abs(cj(i_coeff)));
    idx = i_coeff(idx); % Translate the index to an absolute index
    
    % Invert the information matrix at the first iteration, later only update its value on
    % the basis of the previous inverted one
    if k == 1
        M = pinv(Psi(:, idx)' * Psi(:, idx));
    else
        x = (Psi(:, a_coeff))' * Psi(:, idx);
        r = (Psi(:, idx))' * Psi(:, idx);
        % Update the information matrix based on the last calculated
        % inverse
        M = aPC_blockwise_inverse(M, x, x', r); 
        
        % If the resulting matrix is singular, throw out a warning and use pinv
        if any(~isfinite(M(:)))
            try
                M = pinv(Psi(:, [a_coeff, idx])' * Psi(:, [a_coeff, idx]));
            catch me
                if DisplayLevel > 1
                    warning('Singular design matrix. Skipping the current basis element and removing it from the candidates to improve stability');
                end
                i_coeff(i_coeff == idx) = [];
                M = pinv(Psi(:, a_coeff)' * Psi(:, a_coeff));
                continue;
            end
        end
    end
    
    % Update the set of active predictors with the newfound idx
    a_coeff = [a_coeff, idx];
    
    % Now we get the vector of correlation signs
    s = sign(cj(a_coeff));
    s(~s) = 1;  % Set the null signs to positive
    
    %% Calculate gamma, w, and u based on LARS
    c = 1 / sqrt((s' * M) * s);
    w = c * M * s;
    u = Psi(:, a_coeff) * w;
    aj = Psi' * u;
    
    % Calculate gamma, based on LARS equations
    if k < nvars
        tmp = [(C - cj(i_coeff)) ./ (c - aj(i_coeff)); (C + cj(i_coeff)) ./ (c + aj(i_coeff))];
        gamma = min(tmp(tmp > 0));
        if isempty(gamma)
            gamma = 0;
            warning('Warning: numerical instability!! Gamma for LAR iteration %d was set to 0 to prevent crashes.', k);
        end
    else
        gamma = C / c;
    end
    
    % Remove the coefficient from the candidate predictors
    i_coeff(i_coeff == idx) = [];
    
    % Now update the residual
    mu = mu + gamma * u;
    
    % Finally update the coefficients in the correct direction to yield equicorrelated
    coeff_array(k + 1, a_coeff) = coeff_array(k, a_coeff) + gamma * w';
    
    %% Adaptive LARS: Modified Leave One Out error estimate Q^2
    if ~hybrid_loo
        loo = aPC_PCE_loo_error(Psi(:, a_coeff), M, Y, coeff_array(k + 1, a_coeff)', modified_loo, modi_diag);
    else
        loo = aPC_PCE_loo_error(Psi(:, a_coeff), M, Y, [], modified_loo, modi_diag);
    end
    if loo < 0
        warning('Leave one out error negative!!');
    end
        
    loo_scores(k + 1) = loo;
    a_scores(k + 1) = 1 - loo;
    
    % Stop the iterations if the error increases again
    mm = round(nvars * 0.1);
    mm = max(mm, 100);
    mm = min(mm, nvars);
    
    % Update the current best score
    if loo < refscore
        refscore = loo;
    end
    
    if k > mm
        if (loo_scores(k - mm) <= refscore) && early_stop
            if DisplayLevel > 1
                fprintf('Early stop at coefficient %d/%d \n', k - mm, P);
            end
            break;
        end
    end
end

nz_idx = abs(coeff_array(k, :)) > 0;

% Get the best score in the current array
if ~no_selection
    [maxScore, k] = max(a_scores);
else
    maxScore = 1 - loo;
    k = length(a_scores);
end

%% Assigning the coefficients with the best candidate basis via OLS (Hybrid LARS)
% Rescale back the Psi matrix to the original shape
if normalize_columns
    Psi(:, nznorm_indices) = Psi(:, nznorm_indices) .* repmat(normPsi(nznorm_indices), N, 1);
end

if bool_const_and_center
    Psi = bsxfun(@plus, Psi, mu_Psi);
    Y = Y + mu_Y;
    nz_idx(constindices(1)) = 1;
end

% Now we assign the coefficients, either through an extra hybrid_lars
% iteration, or by rescaling of the (non-hybrid) LARS solution.
coefficients = zeros(P, 1);

% ----------------------------------------------------
% NEW: choose basis set by option
%   - use_lars_order = 1 → 使用 LARS 顺序 a_coeff(1:k)
%   - use_lars_order = 0 → 使用原始 nz_idx 行为
% ----------------------------------------------------
if use_lars_order
    % ---- 使用 LARS 排序 ----
    if bool_const_and_center
        % 有中心化 ⇒ 需要手动加常数项
        active = [constindices(1), a_coeff(1:(k - 1))];
    else
        % 无中心化 ⇒ Psi 中已经包含常数项，不需要再加
        active = a_coeff(1:(k - 1));
    end
else
    % ---- 使用原来 nz_idx 行为 ----
    active = find(nz_idx);
end

if hybrid_lars
    ols_results = aPC_OLS(Psi(:, active), Y); % Do not use the covariance option, as Y and Psi are already decorrelated
    coefficients(active) = ols_results.coefficients;
    results.LOO = ols_results.LOO;
    results.normEmpErr = ols_results.normEmpErr;
else
    coefficients(active) = coeff_array(k, active);
    if normalize_columns
        coefficients(nznorm_indices) = coefficients(nznorm_indices) ./ normPsi(nznorm_indices)';
    end
    if bool_const_and_center
        coefficients(constindices(1)) = mean(Y - Psi * coefficients) / constval(1);
    end
    [results.LOO, results.normEmpErr, results.optErrorParams] = aPC_PCE_loo_error(Psi(:, nz_idx), pinv(Psi(:, nz_idx)' * Psi(:, nz_idx)), Y, coefficients(nz_idx), 1);
end

% if hybrid_lars
%     ols_results = aPC_OLS(Psi(:, nz_idx), Y); % Do not use the covariance option, as Y and Psi are already decorrelated
%     coefficients(nz_idx) = ols_results.coefficients;
%     results.LOO = ols_results.LOO;
%     results.normEmpErr = ols_results.normEmpErr;
% else
%     coefficients(nz_idx) = coeff_array(k, nz_idx);
%     if normalize_columns
%         coefficients(nznorm_indices) = coefficients(nznorm_indices) ./ normPsi(nznorm_indices)';
%     end
%     if bool_const_and_center
%         coefficients(constindices(1)) = mean(Y - Psi * coefficients) / constval(1);
%     end
%     [results.LOO, results.normEmpErr, results.optErrorParams] = aPC_PCE_loo_error(Psi(:, nz_idx), pinv(Psi(:, nz_idx)' * Psi(:, nz_idx)), Y, coefficients(nz_idx), 1);
% end

if DisplayLevel > 1
    fprintf('LAR basis size: %d/%d\n', sum(nz_idx), P);
end

%% Assign the remaining outputs
results.coeff_array = coeff_array;
results.max_score = maxScore;
results.coefficients = coefficients;
results.a_scores = a_scores;
results.loo_scores = loo_scores;
results.best_basis_index = k;
results.nz_idx = nz_idx;


if bool_const_and_center
    results.lars_idx = [constindices(1) a_coeff(1:(k - 1))];
else
    results.lars_idx = a_coeff(1:(k - 1));
end

results.LOO_lars = 1 - maxScore;

results.method = 'LARS';
results.optErrorParams = ols_results.optErrorParams;
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