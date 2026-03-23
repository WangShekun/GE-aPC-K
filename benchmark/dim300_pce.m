function y = dim300_pce(xx)
% -------------------------------------------------------------------------
% dim300_pce — PCE-friendly 300D benchmark (low-order, mildly coupled)
%
% Purpose:
%   - High-dimensional (d=300) benchmark that is "PCE-friendly":
%     * dominated by low-order terms (<=3)
%     * includes mild coupling (pairwise neighbor interactions)
%     * controllable effective dimension via weights
%
% Form (conceptual):
%   y = c0
%     + sum_i w_i * ( a1*x_i + a2*x_i^2 + a3*x_i^3 )
%     + b2 * sum_i w_i*w_{i+1} * x_i*x_{i+1}
%     + b3 * sum_i w_i*w_{i+1} * (x_i^2*x_{i+1} + x_i*x_{i+1}^2)
%
% Input:
%   xx : N x d (recommend d=300, but works for any d>=3)
% Output:
%   y  : N x 1
% -------------------------------------------------------------------------

x = xx;
[~, d] = size(x);
if d < 3
    error('dim300_pce expects d >= 3.');
end

%% ===== Tunable knobs (you can modify these) =====
tau = 60;      % weight decay length; larger -> higher effective dimension
a1  = 1.0;     % linear strength
a2  = 0.15;    % quadratic strength
a3  = 0.02;    % cubic strength
b2  = 0.25;    % pairwise (2nd-order) coupling strength
b3  = 0.05;    % pairwise (3rd-order) coupling strength
c0  = -10;     % constant shift
% ===============================================

% weights: w_1 >= w_2 >= ... (controls effective dimension)
idx = 0:(d-1);
w   = exp(-idx./tau);           % 1 x d
w2  = w(1:end-1) .* w(2:end);   % 1 x (d-1)

% 1D low-order part
t1 = a1*x + a2*(x.^2) + a3*(x.^3);           % N x d
term1 = t1 * w.';                             % N x 1

% neighbor interactions (2nd order)
term2 = b2 * sum( (x(:,1:end-1).*x(:,2:end)) .* w2, 2 );

% neighbor interactions (3rd order, symmetric)
term3 = b3 * sum( (x(:,1:end-1).^2 .* x(:,2:end) + x(:,1:end-1).*x(:,2:end).^2) .* w2, 2 );

y = c0 + term1 + term2 + term3;

end
