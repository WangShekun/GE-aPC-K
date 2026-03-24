function dYdX = ishigami_grad(X, a, b)
% Gradient of ishigami(X,a,b) with default a=7, b=0.1.

    if nargin < 2 || isempty(a)
        a = 7;
    end
    if nargin < 3 || isempty(b)
        b = 0.1;
    end

    x1 = X(:, 1);
    x2 = X(:, 2);
    x3 = X(:, 3);

    dYdX = [ ...
        cos(x1) .* (1 + b .* x3.^4), ...
        2 * a .* sin(x2) .* cos(x2), ...
        4 * b .* x3.^3 .* sin(x1) ...
    ];
end
