function dYdX = dim100_grad(X)
% Analytic gradient of dim100(X).

    [N, d] = size(X);
    idx = 1:d;
    idx_row = repmat(idx, N, 1);

    numer = 2 .* X + 4 .* X.^3;
    denom = X.^2 + X.^4;

    dYdX = (idx_row .* (3 .* X.^2 + numer ./ (3 .* denom) - 5)) ./ d;

    dYdX(:, 1) = dYdX(:, 1) + X(:, 2).^2;
    dYdX(:, 2) = dYdX(:, 2) + 2 .* X(:, 1) .* X(:, 2) + X(:, 4);
    dYdX(:, 3) = dYdX(:, 3) - X(:, 5);
    dYdX(:, 4) = dYdX(:, 4) + X(:, 2);
    dYdX(:, 5) = dYdX(:, 5) - X(:, 3);
    dYdX(:, 50) = dYdX(:, 50) + X(:, 54).^2;
    dYdX(:, 51) = dYdX(:, 51) + 1;
    dYdX(:, 54) = dYdX(:, 54) + 2 .* X(:, 50) .* X(:, 54);
end
