function dYdX = dixonpr50_grad(X)
% Analytic gradient of dixonpr50(X).

    [N, d] = size(X);
    if d ~= 50
        error('dixonpr50_grad expects N x 50 input.');
    end

    dYdX = zeros(N, d);

    t2 = 2 * X(:, 2).^2 - X(:, 1);
    dYdX(:, 1) = 2 * (X(:, 1) - 1) - 4 * t2;

    for j = 2:(d - 1)
        tj = 2 * X(:, j).^2 - X(:, j - 1);
        tj1 = 2 * X(:, j + 1).^2 - X(:, j);
        dYdX(:, j) = 8 * j * X(:, j) .* tj - 2 * (j + 1) * tj1;
    end

    td = 2 * X(:, d).^2 - X(:, d - 1);
    dYdX(:, d) = 8 * d * X(:, d) .* td;
end
