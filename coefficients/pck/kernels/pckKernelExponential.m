function Kff = pckKernelExponential(delta, theta)
    % Exponential kernel values for APCK.
    % We intentionally do not use this kernel in GEAPCK because it is not
    % differentiable at zero.
    theta = reshape(theta, 1, 1, []);
    q = sum(abs(delta) ./ theta, 3);
    Kff = exp(-q);
end
