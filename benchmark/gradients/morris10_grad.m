function dYdX = morris10_grad(X)
% Analytic gradient of morris10(X).

    [N, M] = size(X);
    if M ~= 20
        error('morris10_grad expects N x 20 input.');
    end

    W = 2 * (X - 0.5);
    special = [3 5 7];
    Usp = X(:, special);
    W(:, special) = 2 * (1.1 * Usp ./ (Usp + 0.1) - 0.5);

    dW = 2 * ones(N, M);
    dW(:, special) = 0.22 ./ (Usp + 0.1).^2;

    dYdW = zeros(N, M);

    % First-order terms
    for i = 1:20
        if i <= 10
            bi = 20;
        else
            bi = (-1)^i;
        end
        dYdW(:, i) = dYdW(:, i) + bi;
    end

    % Second-order terms
    for i = 1:19
        for j = i+1:20
            if (i <= 6) && (j <= 6)
                bij = -15;
            else
                bij = (-1)^(i + j);
            end
            dYdW(:, i) = dYdW(:, i) + bij .* W(:, j);
            dYdW(:, j) = dYdW(:, j) + bij .* W(:, i);
        end
    end

    % Third-order terms
    for i = 1:18
        for j = i+1:19
            for k = j+1:20
                if (i <= 5) && (j <= 5) && (k <= 5)
                    bijk = -10;
                else
                    bijk = 0;
                end
                if bijk ~= 0
                    dYdW(:, i) = dYdW(:, i) + bijk .* W(:, j) .* W(:, k);
                    dYdW(:, j) = dYdW(:, j) + bijk .* W(:, i) .* W(:, k);
                    dYdW(:, k) = dYdW(:, k) + bijk .* W(:, i) .* W(:, j);
                end
            end
        end
    end

    % Fourth-order term: 5 * W1 * W2 * W3 * W4
    c4 = 5;
    dYdW(:, 1) = dYdW(:, 1) + c4 .* W(:, 2) .* W(:, 3) .* W(:, 4);
    dYdW(:, 2) = dYdW(:, 2) + c4 .* W(:, 1) .* W(:, 3) .* W(:, 4);
    dYdW(:, 3) = dYdW(:, 3) + c4 .* W(:, 1) .* W(:, 2) .* W(:, 4);
    dYdW(:, 4) = dYdW(:, 4) + c4 .* W(:, 1) .* W(:, 2) .* W(:, 3);

    dYdX = dYdW .* dW;
end
