function [Y] = sobolG(X)
% -------------------------------------------------------------------------
% Sobol' g-function
%
% Input:
%   X : N × n   (each row is one sample, X_i ~ U(0,1))
%
% Output:
%   Y : N × 1
% -------------------------------------------------------------------------

[N, n] = size(X);

% coefficients a_i
a = 500 * ones(1, n);
a(1:2) = 1;

% threshold
b = 0.35;

% Sobol' g-function
G = (abs(4*X - 2) + a) ./ (1 + a);   % N × n
Y = prod(G, 2) - b;                 % N × 1

end
