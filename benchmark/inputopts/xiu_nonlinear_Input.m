function InputOpts = xiu_nonlinear_Input()
% Input definition for Xiu nonlinear algebraic problem
%
% ---- CHANGE PARAMETERS HERE ----
d      = 200;    % dimension (paper used 4; you may test 40 / 100 / 200)
mu_x   = 0.0;   % mean of X_i (0 or 1 in the paper)
sigma_x = 0.2;  % standard deviation of X_i
% --------------------------------

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Gaussian';
    InputOpts.Marginals(ii).Parameters = [mu_x sigma_x];
end
end
