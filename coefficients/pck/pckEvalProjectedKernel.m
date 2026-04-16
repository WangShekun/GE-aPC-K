function [Kff, Kfg_modes, Kgg_modes] = pckEvalProjectedKernel(X1, X2, theta, corr_type, V_r)
    delta = localPairwiseDelta(X1, X2);
    inv_theta2 = 1 ./ (theta(:) .^ 2);
    proj_delta = localProjectedDelta(delta, V_r, inv_theta2);
    proj_metric = V_r' * bsxfun(@times, V_r, inv_theta2);

    switch lower(corr_type)
        case 'gaussian'
            q = sum((delta ./ reshape(theta(:), 1, 1, [])).^2, 3);
            Kff = exp(-0.5 * q);
            Kfg_modes = localGaussianProjectedKfg(proj_delta, Kff);
            if nargout < 3
                return;
            end
            Kgg_modes = localGaussianProjectedKgg(proj_delta, proj_metric, Kff);
        case 'matern32'
            a = sqrt(3);
            radial = sqrt(sum((delta ./ reshape(theta(:), 1, 1, [])).^2, 3));
            E = exp(-a * radial);
            Kff = (1 + a * radial) .* E;
            Kfg_modes = localMatern32ProjectedKfg(proj_delta, E);
            if nargout < 3
                return;
            end
            Kgg_modes = localMatern32ProjectedKgg(proj_delta, proj_metric, radial, E);
        otherwise
            error('aPC:UnsupportedGECorrelation', ...
                'GEAPCK low-rank projected kernel requires supported differentiable kernel: %s', corr_type);
    end
end

function delta = localPairwiseDelta(X1, X2)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    delta = reshape(X1, n1, 1, []) - reshape(X2, 1, n2, []);
end

function proj_delta = localProjectedDelta(delta, V_r, inv_theta2)
    n1 = size(delta, 1);
    n2 = size(delta, 2);
    r = size(V_r, 2);
    d = size(delta, 3);
    proj_delta = zeros(n1, n2, r);
    for k = 1:r
        acc = zeros(n1, n2);
        for a = 1:d
            acc = acc + (V_r(a, k) * inv_theta2(a)) * delta(:, :, a);
        end
        proj_delta(:, :, k) = acc;
    end
end

function Kfg_modes = localGaussianProjectedKfg(proj_delta, Kff)
    r = size(proj_delta, 3);
    Kfg_modes = cell(1, r);
    for k = 1:r
        Kfg_modes{k} = proj_delta(:, :, k) .* Kff;
    end
end

function Kgg_modes = localGaussianProjectedKgg(proj_delta, proj_metric, Kff)
    r = size(proj_delta, 3);
    Kgg_modes = cell(r, r);
    for k = 1:r
        for l = 1:r
            Kgg_modes{k, l} = (proj_metric(k, l) - proj_delta(:, :, k) .* proj_delta(:, :, l)) .* Kff;
        end
    end
end

function Kfg_modes = localMatern32ProjectedKfg(proj_delta, E)
    r = size(proj_delta, 3);
    Kfg_modes = cell(1, r);
    for k = 1:r
        Kfg_modes{k} = 3 * E .* proj_delta(:, :, k);
    end
end

function Kgg_modes = localMatern32ProjectedKgg(proj_delta, proj_metric, radial, E)
    r = size(proj_delta, 3);
    safe_radial = radial;
    safe_radial(safe_radial == 0) = 1;
    Kgg_modes = cell(r, r);
    for k = 1:r
        for l = 1:r
            term = -(sqrt(3) * proj_delta(:, :, k) .* proj_delta(:, :, l)) ./ safe_radial;
            term(radial == 0) = 0;
            Kgg_modes{k, l} = 3 * E .* (proj_metric(k, l) + term);
        end
    end
end
