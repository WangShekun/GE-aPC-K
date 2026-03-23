function y = xiu_nonlinear(xx)
% -------------------------------------------------------------------------
% Nonlinear algebraic test function (Xiu 2007; Xu & Rahman 2004)
%
% Z = sqrt( 1 + 0.5 * X*X^T ) * log( 0.5 + X*X^T )
%
% where:
%   X*X^T = sum_{i=1}^d x_i^2
%
% Input:
%   xx : N x d matrix, rows are samples
%
% Output:
%   y  : N x 1 model responses
%
% Properties:
%   - Strong nonlinearity (quadratic + logarithmic)
%   - All variables contribute equally
%   - Effective dimension ≈ d
%   - Classical UQ benchmark (Hermite PC often used)
% -------------------------------------------------------------------------

x = xx;

% Quadratic form: X X^T
q = sum(x.^2, 2);

% Model response
y = sqrt(1 + 0.5*q) .* log(0.5 + q);

end
