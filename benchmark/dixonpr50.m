function y = dixonpr50(X)
% dixonpr50 - 50D Dixon-Price benchmark evaluated row-wise.

    if size(X, 2) ~= 50
        error('dixonpr50 expects N x 50 input.');
    end

    N = size(X, 1);
    y = (X(:, 1) - 1).^2;
    for ii = 2:50
        y = y + ii * (2 * X(:, ii).^2 - X(:, ii - 1)).^2;
    end

    y = reshape(y, N, 1);
end
