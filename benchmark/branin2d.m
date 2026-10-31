function y = branin2d(X)
%BRANIN2D Classic 2D multimodal benchmark function.
%   Domain: x1 in [-5, 10], x2 in [0, 15]

x1 = X(:, 1);
x2 = X(:, 2);

a = 1.0;
b = 5.1 / (4 * pi^2);
c = 5 / pi;
r = 6.0;
s = 10.0;
t = 1.0 / (8 * pi);

y = a .* (x2 - b .* x1.^2 + c .* x1 - r).^2 + s .* (1 - t) .* cos(x1) + s;
end
