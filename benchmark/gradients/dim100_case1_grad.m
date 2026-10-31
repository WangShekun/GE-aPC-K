function dYdX = dim100_case1_grad(X)
% Analytic gradient of dim100_case1(X).

[N, d] = size(X);
if d ~= 100
    error('dim100_case1_grad expects a N x 100 input matrix.');
end

dYdX = 1 - exp(X - 2) .* (sin(X) + cos(X));

dYdX(:, 1) = dYdX(:, 1) + 30 .* X(:, 2);
dYdX(:, 2) = dYdX(:, 2) + 30 .* X(:, 1) + X(:, 3);

for j = 3:99
    dYdX(:, j) = dYdX(:, j) + X(:, j - 1) + X(:, j + 1);
end

dYdX(:, 100) = dYdX(:, 100) + X(:, 99);
end
