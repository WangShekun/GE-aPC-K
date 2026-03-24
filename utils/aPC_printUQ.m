function aPC_printUQ(UQ)
% aPC_printUQ Print a normalized uncertainty quantification report.

    width = 84;
    divider = repmat('-', 1, width);
    banner = localBanner('UQ REPORT', width);

    fprintf('\n%s\n', banner);

    localSection('Core Statistics', width);
    localKV('Mean (E[Y])', localMetric(UQ, 'mean'));
    localKV('Variance (Var[Y])', localMetric(UQ, 'var'));
    localKV('Std deviation', localMetric(UQ, 'std'));
    fprintf('%s\n', divider);

    if any(isfield(UQ, {'CI95', 'skew', 'kurt', 'VaR95', 'CVaR95'}))
        localSection('Extended Statistics', width);
        if isfield(UQ, 'CI95') && numel(UQ.CI95) == 2
            localKV('95% credible interval', sprintf('[%.6e, %.6e]', UQ.CI95(1), UQ.CI95(2)));
        end
        if isfield(UQ, 'skew')
            localKV('Skewness', sprintf('%.6e', UQ.skew));
        end
        if isfield(UQ, 'kurt')
            localKV('Kurtosis', sprintf('%.6e', UQ.kurt));
        end
        if isfield(UQ, 'VaR95')
            localKV('VaR_{95}', sprintf('%.6e', UQ.VaR95));
        end
        if isfield(UQ, 'CVaR95')
            localKV('CVaR_{95}', sprintf('%.6e', UQ.CVaR95));
        end
        fprintf('%s\n', divider);
    end

    fprintf('%s\n', banner);
end

function localSection(title, width)
    fprintf('%s\n', localBanner(title, width));
end

function line = localBanner(title, width)
    inner = sprintf(' %s ', title);
    pad_total = max(0, width - strlength(inner));
    left = floor(pad_total / 2);
    right = pad_total - left;
    line = [repmat('=', 1, left), inner, repmat('=', 1, right)];
end

function localKV(key, val)
    fprintf('%-28s : %s\n', key, val);
end

function val = localMetric(S, field)
    if isfield(S, field) && ~isempty(S.(field))
        val = sprintf('%.6e', S.(field));
    else
        val = '-';
    end
end
