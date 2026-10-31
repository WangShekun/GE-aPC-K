function y = rastrigin2d(X)
%RASTRIGIN2D Classic 2D highly multimodal benchmark.
%   Domain: x1, x2 in [-5.12, 5.12]

x1 = X(:, 1);
x2 = X(:, 2);
A = 10;

y = 2 * A + (x1.^2 - A .* cos(2 * pi .* x1)) + (x2.^2 - A .* cos(2 * pi .* x2));
end
