function y = rackwitz(xx)
% -------------------------------------------------------------------------
% Rackwitz high-dimensional limit-state function
%
% G(x) = n + 3*sigma*sqrt(n) - sum_{i=1}^n x_i
%
% References:
%   Rackwitz (2000)
%   Bourinet et al. (2011)
%   Used in SIR-SPCE high-dimensional reliability analysis
%
% Input:
%   xx : N x d matrix (x_i are i.i.d. Lognormal)
%
% Output:
%   y  : N x 1 vector (limit-state value)
%
% Notes:
%   - All variables contribute equally
%   - Effective dimension ≈ d (no reducible structure)
%   - Excellent benchmark for testing robustness of surrogate models
% -------------------------------------------------------------------------

x = xx;
[N, d] = size(x);

sigma = 0.2;

y = d + 3*sigma*sqrt(d) - sum(x, 2);
end
