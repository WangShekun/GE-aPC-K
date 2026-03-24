function aPC_printSensitivity(SA, ModelOpts)
% aPC_printSensitivity Print a normalized Sobol sensitivity report.

    M = ModelOpts.num_vars;
    names = localVarNames(M);
    width = 84;
    divider = repmat('-', 1, width);
    banner = localBanner('SENSITIVITY REPORT', width);

    fprintf('\n%s\n', banner);

    localSection('Total-Order Indices', width);
    localPrintTable(names, SA.S_total);
    fprintf('%s\n', divider);

    localSection('First-Order Indices', width);
    localPrintTable(names, SA.S_first);
    fprintf('%s\n', divider);

    localSection('Second-Order Indices', width);
    [second_names, second_vals] = localSecondOrder(SA.S_second, names);
    localPrintTable(second_names, second_vals);
    fprintf('%s\n', divider);

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

function localPrintTable(names, vals)
    if isempty(names)
        fprintf('[none]\n');
        return;
    end

    for i = 1:numel(names)
        fprintf('%-12s', names{i});
    end
    fprintf('\n');
    for i = 1:numel(vals)
        fprintf('%-12.6f', vals(i));
    end
    fprintf('\n');
end

function [names, vals] = localSecondOrder(S_second, base_names)
    M = numel(base_names);
    names = {};
    vals = [];
    for i = 1:M
        for j = i+1:M
            names{end+1} = sprintf('%s%s', base_names{i}, base_names{j}); %#ok<AGROW>
            vals(end+1) = S_second(i, j); %#ok<AGROW>
        end
    end
end

function names = localVarNames(M)
    names = arrayfun(@(i) sprintf('X%d', i), 1:M, 'UniformOutput', false);
end
