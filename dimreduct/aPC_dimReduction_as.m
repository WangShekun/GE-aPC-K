function [X_red, ReduceModel] = aPC_dimReduction_as(U, dYdU, ncomp)
% Active Subspace reduction using gradient covariance only.

    if isempty(dYdU)
        error('aPC:ReductionGradientRequired', 'as requires gradients in U-space.');
    end

    [~, M] = size(U);
    if numel(ncomp) ~= 1 || ncomp <= 0 || floor(ncomp) ~= ncomp || ncomp > M
        error('aPC:InvalidASDim', 'Active subspace red_dim must be an integer in [1, size(U,2)].');
    end

    C = (dYdU' * dYdU) / max(size(dYdU, 1), 1);
    C = (C + C') * 0.5;
    [V, D] = eig(C);
    [eigvals, idx] = sort(diag(D), 'descend');
    A = V(:, idx(1:ncomp));

    X_red = U * A;

    params = struct();
    params.A = A;
    params.eigvals = eigvals;

    ReduceModel = struct();
    ReduceModel.method = 'as';
    ReduceModel.params = params;
    ReduceModel.transform = @(U_new) U_new * params.A;
end
