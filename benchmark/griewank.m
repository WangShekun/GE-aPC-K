function y = griewank(xx)
% -------------------------------------------------------------------------
% Griewank function (high-dimensional form)
%
% y(x) = sum_{i=1}^d (x_i^2 / 4000)
%        - prod_{i=1}^d cos( x_i / sqrt(i) ) + 1
%
% Domain:
%   x_i in [-10, 10]
%
% Properties:
%   - Classical benchmark (optimization, surrogate, DR literature)
%   - Nonlinear and oscillatory
%   - Weak but global coupling via product term
%   - Effective dimension: medium-to-high (grows with d)
%
% Input:
%   xx : N x d matrix
%
% Output:
%   y  : N x 1 vector
% -------------------------------------------------------------------------

x = xx;
[N, d] = size(x);

% Term 1: quadratic sum
term1 = sum(x.^2, 2) / 4000;

% Term 2: cosine product
idx   = 1:d;
term2 = prod( cos( x ./ sqrt(idx) ), 2 );

% Griewank output
y = term1 - term2 + 1;

end
