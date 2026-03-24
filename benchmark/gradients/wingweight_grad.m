function dYdX = wingweight_grad(X)
% Analytic gradient of wingweight(X).

    Sw = X(:, 1);
    Wfw = X(:, 2);
    A = X(:, 3);
    LamDeg = X(:, 4);
    q = X(:, 5);
    lam = X(:, 6);
    tc = X(:, 7);
    Nz = X(:, 8);
    Wdg = X(:, 9);
    Wp = X(:, 10);

    Lam = LamDeg * (pi / 180);
    cLam = cos(Lam);
    tLam = tan(Lam);

    fact1 = 0.036 .* Sw.^0.758 .* Wfw.^0.0035;
    fact2 = (A ./ (cLam.^2)).^0.6;
    fact3 = q.^0.006 .* lam.^0.04;
    fact4 = (100 .* tc ./ cLam).^(-0.3);
    fact5 = (Nz .* Wdg).^0.49;

    core = fact1 .* fact2 .* fact3 .* fact4 .* fact5;

    dYdX = zeros(size(X));

    dYdX(:, 1) = core .* (0.758 ./ Sw) + Wp;
    dYdX(:, 2) = core .* (0.0035 ./ Wfw);
    dYdX(:, 3) = core .* (0.6 ./ A);
    dYdX(:, 4) = core .* (0.9 .* tLam) .* (pi / 180);
    dYdX(:, 5) = core .* (0.006 ./ q);
    dYdX(:, 6) = core .* (0.04 ./ lam);
    dYdX(:, 7) = core .* (-0.3 ./ tc);
    dYdX(:, 8) = core .* (0.49 ./ Nz);
    dYdX(:, 9) = core .* (0.49 ./ Wdg);
    dYdX(:, 10) = Sw;
end
