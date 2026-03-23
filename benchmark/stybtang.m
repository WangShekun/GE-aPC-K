function y = stybtang(xx)
% -------------------------------------------------------------------------
% STYBLINSKI–TANG FUNCTION (vectorized, high-dimensional form)
%
% Classic benchmark function used in:
%   - global optimization
%   - surrogate modeling
%   - dimensionality reduction studies
%
% Definition:
%   y(x) = 1/2 * sum_{i=1}^d ( x_i^4 - 16 x_i^2 + 5 x_i )
%
% Characteristics:
%   - Non-convex, multimodal
%   - Additive structure (no explicit interactions)
%   - Effective dimension ≈ d (not strongly reducible)
%
% Input:
%   xx : N x d matrix
%
% Output:
%   y  : N x 1 vector
% -------------------------------------------------------------------------

x = xx;

term = x.^4 - 16*x.^2 + 5*x;
y = 0.5 * sum(term, 2);

end
