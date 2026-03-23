function [Y] = zhou98(X)
% -------------------------------------------------------------------------
% ZHOU (1998) function
%
% Input:
%   X : N × d   (each row is one sample)
%
% Output:
%   Y : N × 1
% -------------------------------------------------------------------------

[N, d] = size(X);

% Two Gaussian centers
c1 = 1/3;
c2 = 2/3;

% Scaled distances
X1 = 10 * (X - c1);     % N × d
X2 = 10 * (X - c2);     % N × d

% Squared norms (row-wise)
r1 = sum(X1.^2, 2);     % N × 1
r2 = sum(X2.^2, 2);     % N × 1

% Gaussian densities
phi1 = (2*pi)^(-d/2) .* exp(-0.5 * r1);
phi2 = (2*pi)^(-d/2) .* exp(-0.5 * r2);

% Final output
Y = (10^d)/2 .* (phi1 + phi2);

end
