function grid = aPC_build_search_grid(ModelOpts)

search_fields = aPC_list_search_fields();

keynames = {};
keyvals  = {};

for i = 1:numel(search_fields)
    fn = search_fields{i};

    try
        subs = strsplit(fn,'.');
        val = getfield(ModelOpts, subs{:});
    catch
        continue;
    end

    % scalar → ignore
    if isnumeric(val) && isscalar(val)
        continue;
    end

    % vector numeric → convert to cell list
    if isnumeric(val)
        keynames{end+1} = fn;
        keyvals{end+1}  = num2cell(val(:)');
        continue;
    end

    % cell list → direct search space
    if iscell(val)
        if numel(val) > 1      % 多值 → 搜索
            keynames{end+1} = fn;
            keyvals{end+1}  = val(:)';
        end
        continue;
    end
end

% no search → return single model
if isempty(keynames)
    grid = ModelOpts;
    return;
end

lens = cellfun(@numel,keyvals);
idx  = ones(1,numel(keynames));
configs = {};

while true
    Mo = ModelOpts;

    for k = 1:numel(keynames)
        subs = strsplit(keynames{k},'.');
        Mo = setfield(Mo, subs{:}, keyvals{k}{idx(k)});
    end

    configs{end+1} = Mo;

    % increment logic
    j = numel(idx);
    while j>=1
        idx(j)=idx(j)+1;
        if idx(j) <= lens(j)
            break;
        end
        idx(j)=1;
        j=j-1;
    end
    if j <= 0, break; end
end

grid = [configs{:}];
end
