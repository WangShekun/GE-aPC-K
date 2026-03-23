function aPC_printUQ(UQ)

line  = repmat('-', 1, 72);
line2 = repmat('=', 1, 72);

fprintf('\n%s\n', line2);
fprintf('aPC-based Uncertainty Quantification\n');
fprintf('%s\n', line2);

%% ----- Analytic statistics (always available if computed) -----
if isfield(UQ,'mean')
    fprintf('  Mean (E[Y])             : %12.6e\n', UQ.mean);
end

if isfield(UQ,'var')
    fprintf('  Variance (Var[Y])       : %12.6e\n', UQ.var);
end

if isfield(UQ,'std')
    fprintf('  Std deviation (Std[Y])  : %12.6e\n', UQ.std);
end

fprintf('%s\n', line);

%% ----- Extended statistics (conditional) -----
if isfield(UQ,'CI95')
    fprintf('  95%% credible interval   : [%12.6e, %12.6e]\n', ...
            UQ.CI95(1), UQ.CI95(2));
end

if isfield(UQ,'skew')
    fprintf('  Skewness                : %12.6e\n', UQ.skew);
end

if isfield(UQ,'kurt')
    fprintf('  Kurtosis                : %12.6e\n', UQ.kurt);
end

if isfield(UQ,'VaR95')
    fprintf('  VaR_{95}                : %12.6e\n', UQ.VaR95);
end

if isfield(UQ,'CVaR95')
    fprintf('  CVaR_{95}               : %12.6e\n', UQ.CVaR95);
end

fprintf('%s\n\n', line2);

end
