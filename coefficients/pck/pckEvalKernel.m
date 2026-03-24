function [Kff, Kfg_blocks, Kgg_blocks] = pckEvalKernel(X1, X2, theta, corr_type)
    % Shared kernel dispatcher. Optional outputs expose derivative blocks used by
    % GEAPCK; APCK only needs the value block Kff.
    delta = localPairwiseDelta(X1, X2);

    switch lower(corr_type)
        case 'gaussian'
            [Kff, Kfg_blocks, Kgg_blocks] = pckKernelGaussian(delta, theta);
        case 'exponential'
            Kff = pckKernelExponential(delta, theta);
            Kfg_blocks = {};
            Kgg_blocks = {};
        case 'matern32'
            [Kff, Kfg_blocks, Kgg_blocks] = pckKernelMatern32(delta, theta);
        otherwise
            error('aPC:UnsupportedCorrelation', 'Unsupported corr type: %s', corr_type);
    end
end

function delta = localPairwiseDelta(X1, X2)
    % delta(i,j,:) = X1(i,:) - X2(j,:)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    delta = reshape(X1, n1, 1, []) - reshape(X2, 1, n2, []);
end
