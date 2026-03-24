function [Kff, Kfg_blocks, Kgg_blocks] = pckKernelMatern32(delta, theta)
    % Matern 3/2 kernel plus analytic derivative blocks for GEAPCK.
    theta = reshape(theta, 1, 1, []);
    theta2 = reshape(theta.^2, 1, 1, []);
    a = sqrt(3);
    r = sqrt(sum((delta ./ theta).^2, 3));
    E = exp(-a * r);
    Kff = (1 + a * r) .* E;

    if nargout < 2
        return;
    end

    d = size(delta, 3);
    Kfg_blocks = cell(1, d);
    for c = 1:d
        Kfg_blocks{c} = 3 * E .* (delta(:, :, c) ./ theta2(1, 1, c));
    end

    if nargout < 3
        return;
    end

    safe_r = r;
    safe_r(safe_r == 0) = 1;
    Kgg_blocks = cell(d, d);
    for aidx = 1:d
        Da = delta(:, :, aidx);
        inv_a = theta2(1, 1, aidx);
        for bidx = 1:d
            Db = delta(:, :, bidx);
            inv_b = theta2(1, 1, bidx);
            term = -(sqrt(3) * Da .* Db) ./ (inv_a * inv_b .* safe_r);
            term(r == 0) = 0;
            if aidx == bidx
                term = term + 1 / inv_b;
            end
            Kgg_blocks{aidx, bidx} = 3 * E .* term;
        end
    end
end
