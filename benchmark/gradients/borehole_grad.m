function dYdX = borehole_grad(X)
% Analytic gradient of borehole(X).

    rw = X(:, 1);
    r = X(:, 2);
    Tu = X(:, 3);
    Hu = X(:, 4);
    Tl = X(:, 5);
    Hl = X(:, 6);
    L = X(:, 7);
    Kw = X(:, 8);

    deltaH = Hu - Hl;
    log_term = log(r ./ rw);
    B = (2 .* L .* Tu) ./ (log_term .* rw.^2 .* Kw);
    C = Tu ./ Tl;
    A = 1 + B + C;

    numerator = 2 .* pi .* Tu .* deltaH;
    denominator = log_term .* A;
    Y = numerator ./ denominator;

    dYdX = zeros(size(X));

    dLg_rw = -1 ./ rw;
    dLg_r = 1 ./ r;

    dB_rw = B .* (1 ./ (rw .* log_term) - 2 ./ rw);
    dB_r = -B ./ (r .* log_term);
    dB_Tu = B ./ Tu;
    dB_L = B ./ L;
    dB_Kw = -B ./ Kw;

    dC_Tu = 1 ./ Tl;
    dC_Tl = -Tu ./ (Tl.^2);

    dlnD_rw = dLg_rw ./ log_term + dB_rw ./ A;
    dlnD_r = dLg_r ./ log_term + dB_r ./ A;
    dlnD_Tu = (dB_Tu + dC_Tu) ./ A;
    dlnD_Tl = dC_Tl ./ A;
    dlnD_L = dB_L ./ A;
    dlnD_Kw = dB_Kw ./ A;

    dYdX(:, 1) = -Y .* dlnD_rw;
    dYdX(:, 2) = -Y .* dlnD_r;
    dYdX(:, 3) = Y .* (1 ./ Tu - dlnD_Tu);
    dYdX(:, 4) = Y ./ deltaH;
    dYdX(:, 5) = -Y .* dlnD_Tl;
    dYdX(:, 6) = -Y ./ deltaH;
    dYdX(:, 7) = -Y .* dlnD_L;
    dYdX(:, 8) = -Y .* dlnD_Kw;
end
