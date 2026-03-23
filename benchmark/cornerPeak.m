function [Y] = cornerPeak(X)
% -------------------------------------------------------------------------
% Corner-Peak function (from PCE literature)
%
% Input:
%   X : N × d   (each row is one sample, X_i ~ U(0,1))
%
% Output:
%   Y : N × 1
% -------------------------------------------------------------------------

[~, d] = size(X);

% raw coefficients c_k = 1 / k^2
k  = 1:d;
c  = 1 ./ (k.^2);

% normalized coefficients \hat{c}_k
c_hat = 0.25 * (10 / d) * (c / sum(c));

% core function
S = 1 + X * c_hat(:);    % N × 1
Y = S.^(-(d + 1));       % N × 1

end
