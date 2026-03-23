function y_pred = aPC_PCKPredict(Psi_new, U_new, PCKModel)

    trend_idx = PCKModel.trend_idx;
    Fnew = Psi_new(:, trend_idx);

    U_train = PCKModel.U_train;
    theta = PCKModel.theta;
    Rff = localCorr(U_new, U_train, theta, PCKModel.corr);

    if ~isfield(PCKModel, 'use_ge') || ~PCKModel.use_ge
        cov_vec = Rff;
    else
        [Nnew, d] = size(U_new);
        Ntr = size(U_train, 1);
        Cfg = zeros(Nnew, Ntr * d);
        for i = 1:Nnew
            for j = 1:Ntr
                delta = U_new(i, :) - U_train(j, :);
                kij = Rff(i, j);
                for a = 1:d
                    col = (a - 1) * Ntr + j;
                    Cfg(i, col) = (delta(a) / (theta(a)^2)) * kij;
                end
            end
        end
        cov_vec = [Rff, Cfg];
    end

    y_pred = Fnew * PCKModel.beta + cov_vec * PCKModel.alpha;
end

function R = localCorr(X1, X2, theta, corr_type)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    R = zeros(n1, n2);

    for i = 1:n1
        dx = X2 - X1(i, :);
        switch lower(corr_type)
            case 'gaussian'
                q = sum((dx ./ theta).^2, 2);
                R(i, :) = exp(-0.5 * q)';
            case 'exponential'
                q = sum(abs(dx) ./ theta, 2);
                R(i, :) = exp(-q)';
            case 'matern32'
                q = sqrt(sum((dx ./ theta).^2, 2));
                R(i, :) = ((1 + sqrt(3) * q) .* exp(-sqrt(3) * q))';
            otherwise
                error('aPC:UnsupportedCorrelation', 'Unsupported corr type: %s', corr_type);
        end
    end
end
