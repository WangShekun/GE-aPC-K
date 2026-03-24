function [Kff, Kfg_blocks, Kgg_blocks] = pckKernelGaussian(delta, theta)
    % Gaussian kernel plus the first/second derivative blocks required by GEAPCK.
    theta = reshape(theta, 1, 1, []);
    theta2 = reshape(theta.^2, 1, 1, []);
    q = sum((delta ./ theta).^2, 3);
    Kff = exp(-0.5 * q);

    if nargout < 2
        return;
    end

    d = size(delta, 3);
    Kfg_blocks = cell(1, d);
    for a = 1:d
        Kfg_blocks{a} = (delta(:, :, a) ./ theta2(1, 1, a)) .* Kff;
    end

    if nargout < 3
        return;
    end

    Kgg_blocks = cell(d, d);
    for a = 1:d
        Da = delta(:, :, a);
        inv_a = theta2(1, 1, a);
        for b = 1:d
            Db = delta(:, :, b);
            inv_b = theta2(1, 1, b);
            term = -(Da .* Db) / (inv_a * inv_b);
            if a == b
                term = term + 1 / inv_a;
            end
            Kgg_blocks{a, b} = term .* Kff;
        end
    end
end
