function dYdX = rosenbrock_grad(X)
% Analytic gradient of row-wise d-dimensional rosenbrock(X).

    [N, d] = size(X);
    dYdX = zeros(N, d);

    dYdX(:, 1) = -400 .* X(:, 1) .* (X(:, 2) - X(:, 1).^2) + 2 .* (X(:, 1) - 1);
    for j = 2:(d - 1)
        dYdX(:, j) = 200 .* (X(:, j) - X(:, j - 1).^2) - 400 .* X(:, j) .* (X(:, j + 1) - X(:, j).^2) + 2 .* (X(:, j) - 1);
    end
    dYdX(:, d) = 200 .* (X(:, d) - X(:, d - 1).^2);
end
