function [results, ModelOpts, data, ReduceModel] = aPC_Train(data, ModelOpts)
% ------------------------------------------------------------
% 标准化 →（可选降维）→ 多索引生成 → 构造 Ψ → 回归 → LOO 选最优
%
% 支持自动搜索超参数（Order/qnorm/red_dim/kernel_* …）
% 搜索网格由 aPC_build_search_grid.m 自动生成
% ------------------------------------------------------------

%% ---------- 默认设置 ----------
if ~isfield(ModelOpts, 'RegressionMethod') || isempty(ModelOpts.RegressionMethod)
    ModelOpts.RegressionMethod = 'LARS';
end

if ~isfield(ModelOpts, 'qnorm') || isempty(ModelOpts.qnorm)
    ModelOpts.qnorm = 1;
end

ModelOpts = localValidateModelOpts(ModelOpts);

%% ---------- 标准化 ----------
[data, map_U] = aPC_PCE_map_to_U(data.X_train, data);
data.map_U = map_U;

%% ---------- 构造搜索网格 ----------
grid = aPC_build_search_grid(ModelOpts);
total = numel(grid);
parallel_scope = localResolveParallelScope(ModelOpts, total);

if ~strcmp(parallel_scope, 'outer')

    %% ---------- 初始化进度条 ----------
    cnt = 0;
    bar_len = 40;
    localPrintTrainHeader(parallel_scope, total);

    best = struct('score', Inf);

    ReduceModel = [];

    for gi = 1:total

        cnt = cnt + 1;
        pct = cnt / total;

        % ---------- 进度条 ----------
        filled = round(pct * bar_len);
        bar_str = [repmat('#',1,filled), repmat('-',1,bar_len-filled)];

        % 当前候选参数（原始版本）
        Mo = grid(gi);
        Mo.ParallelScope = parallel_scope;

        candidate = aPC_TrainCandidate(data, map_U, Mo);
        Mo_tmp = candidate.ModelOpts;
        data_tmp = candidate.data_model;
        Red_tmp = candidate.ReduceModel;
        res_tmp = candidate.results;
        poly_tmp = candidate.poly;
        univ_tmp = candidate.univ;

        %% ---------- 计算搜索准则（仅 LOO） ----------
        score_tmp = res_tmp.LOO;

        %% ---- 打印进度条信息 ----
        % 打印降维信息
        if isfield(Mo_tmp,'red_dim') && ~isempty(Mo_tmp.red_dim)
            rd_str = num2str(Mo_tmp.red_dim);
        else
            rd_str = '-';
        end

        % 降维方法名（没有就显示 none）
        if isfield(Mo_tmp,'ReductionMethod') && ~isempty(Mo_tmp.ReductionMethod)
            redm = upper(Mo_tmp.ReductionMethod);
        else
            redm = 'NONE';
        end

        % KPCA,SKPCA 显示 kernel 参数
        extra_str = aPC_kernel_print_str(Mo_tmp);

        fprintf('\r%s\n', localFormatTrainProgress(bar_str, pct, cnt, total));
        fprintf('| RD=%s | rd=%s | p=%d | q=%g%s | LOO=%e\n', ...
            redm, rd_str, Mo_tmp.Order, Mo_tmp.qnorm, extra_str, score_tmp);

        % ---------- 更新最优 ----------
    if score_tmp < best.score || ~isfinite(best.score)

            best.LOO        = res_tmp.LOO;
            best.results    = res_tmp;
            best.ModelOpts  = Mo_tmp;
            best.poly       = poly_tmp;
            best.univ       = univ_tmp;
            best.ReduceModel= Red_tmp;
            best.choice     = Mo;
            best.U_red      = data_tmp.U;
            if isfield(data_tmp, 'dY_model')
                best.dYdX_train = data_tmp.dYdX_train;
                best.dY_model = data_tmp.dY_model;
            end
            best.score = score_tmp;
        end

        % aPC_print(res_tmp, Mo_tmp, Mo_tmp.Indices);
    end

    fprintf('\n');

    %% ---------- 写回最终结果 ----------
    results             = best.results;
    ModelOpts           = best.ModelOpts;
    ReduceModel         = best.ReduceModel;

    % ===== 保存最佳降维后的 U =====
    data.U_red = best.U_red;
    if isfield(best, 'dYdX_train')
        data.dYdX_train = best.dYdX_train;
        data.dY_model = best.dY_model;
    end

    results.poly_coeffs = best.poly;
    results.univ_vals   = best.univ;
    results.Indices     = ModelOpts.Indices;
    results.map_U       = map_U;   % 标准化映射

    results.N_train     = size(data.X_train,1);
    if isfield(data,"X_val")
        results.N_test  = size(data.X_val,1);
    else
        results.N_test  = [];
    end

    %% --------- 打印最佳模型 ---------
    mo = ModelOpts;
    
    % --- 降维方法名 ---
    if isfield(mo,'ReductionMethod') && ~isempty(mo.ReductionMethod)
        redm = upper(mo.ReductionMethod);
    else
        redm = 'NONE';
    end
    
    % --- 降维维度 ---
    if isfield(mo,'red_dim') && ~isempty(mo.red_dim)
        rd_str = num2str(mo.red_dim);
    else
        rd_str = 'none';
    end
    
    % --- kernel 信息（仅 KPCA,SKPCA） ---
    extra_str = aPC_kernel_print_str(mo);
    
    fprintf('[aPC_Train] Best Model: RD=%s | rd=%s | p=%d | q=%g%s | LOO=%e\n', ...
            redm, rd_str, mo.Order, mo.qnorm, extra_str, best.score);
    localPrintTrainFooter();

    return;  % 串行版本结束
end

%% ============================================================
%                   并行版本（parallel = true）
% ============================================================
localPrintTrainHeader('outer', total);

% —— 单行进度条 ——
dq = parallel.pool.DataQueue;
cnt = 0; bar_len = 40;
afterEach(dq, @updateBar);

    function updateBar(~)
        cnt = cnt + 1;
        pct = cnt / total;
        filled  = round(pct * bar_len);
        bar_str = [repmat('#',1,filled), repmat('-',1,bar_len-filled)];
        fprintf('\r%s', localFormatTrainProgress(bar_str, pct, cnt, total));
    end

% —— 开并行池 ——
if isempty(gcp('nocreate')), parpool; end

%% —— 并行评估 -------
outs(total) = struct('score', Inf);

parfor gi = 1:total
    Mo = grid(gi);
    Mo.ParallelScope = 'serial';
    outs(gi) = aPC_evaluateCandidate_score(data, Mo);
    send(dq, gi);
end

fprintf('\n');

% —— 最优组合 ——
[~, best_idx] = min([outs.score]);
best_score = outs(best_idx).score;
best_choice   = grid(best_idx);
best_choice.ParallelScope = 'serial';

%% —— 串行重训 ——
candidate = aPC_TrainCandidate(data, map_U, best_choice);
res = candidate.results;
Mo_tmp = candidate.ModelOpts;
Red_tmp = candidate.ReduceModel;
data_tmp = candidate.data_model;
poly_coeffs = candidate.poly;
univ_vals = candidate.univ;
%% —— 写回最终结果 ——
results                 = res;
ModelOpts               = Mo_tmp;
ReduceModel             = Red_tmp;

% ===== 保存最佳降维后的 U（并行版本）=====
data.U_red = data_tmp.U;
if isfield(data_tmp, 'dY_model')
    data.dYdX_train = data_tmp.dYdX_train;
    data.dY_model = data_tmp.dY_model;
end

results.poly_coeffs     = poly_coeffs;
results.univ_vals       = univ_vals;
results.Indices         = ModelOpts.Indices;
results.map_U           = map_U;

results.N_train         = size(data.X_train,1);

if isfield(data,"X_val")
    results.N_test = size(data.X_val,1);
else 
    results.N_test = [];
end
%% --------- 打印最佳模型 ---------
mo = ModelOpts;

% --- 降维方法名 ---
if isfield(mo,'ReductionMethod') && ~isempty(mo.ReductionMethod)
    redm = upper(mo.ReductionMethod);
else
    redm = 'NONE';
end

% --- 降维维度 ---
if isfield(mo,'red_dim') && ~isempty(mo.red_dim)
    rd_str = num2str(mo.red_dim);
else
    rd_str = 'none';
end

% --- kernel 信息（仅 KPCA,SKPCA） ---
extra_str = aPC_kernel_print_str(mo);

fprintf('[aPC_Train] Best Model: RD=%s | rd=%s | p=%d | q=%g%s | LOO=%e\n', ...
        redm, rd_str, mo.Order, mo.qnorm, extra_str, best_score);
localPrintTrainFooter();
end

%% 打印KPCA和SKPCA的核信息
function s = aPC_kernel_print_str(Mo)
% 若是 KPCA/SKPCA，就打印 kernel 信息；否则返回空字符串。

if ~isfield(Mo,'ReductionMethod') || isempty(Mo.ReductionMethod)
    s = '';
    return;
end

m = lower(Mo.ReductionMethod);

% 属于 KPI 族（KPCA + SKPCA）
if any(strcmp(m, {'kpca','skpca_hsic'}))
    s = [' | ' aPC_format_kernel_str(Mo)];
else
    s = '';
end
end

function ModelOpts = localValidateModelOpts(ModelOpts)
    if ~isfield(ModelOpts, 'aPC_Method') || isempty(ModelOpts.aPC_Method)
        ModelOpts.aPC_Method = 'stieltjes';
    end

    if ~isfield(ModelOpts, 'ModelType') || isempty(ModelOpts.ModelType)
        error('aPC:MissingModelType', 'ModelOpts.ModelType is required: apc | apck | geapck | geapck_block.');
    end

    model_type = lower(strtrim(ModelOpts.ModelType));
    valid_model_types = {'apc', 'apck', 'geapck', 'geapck_block'};
    if ~any(strcmp(model_type, valid_model_types))
        error('aPC:InvalidModelType', 'Unsupported ModelType: %s', ModelOpts.ModelType);
    end

    if any(strcmp(model_type, {'apck', 'geapck', 'geapck_block'}))
        if ~isfield(ModelOpts, 'PCKType') || isempty(ModelOpts.PCKType)
            error('aPC:MissingPCKType', 'ModelOpts.PCKType is required for APCK/GEAPCK/GEAPCK_BLOCK.');
        end
        pck_type = lower(strtrim(ModelOpts.PCKType));
        if ~any(strcmp(pck_type, {'sequential', 'optimal'}))
            error('aPC:InvalidPCKType', 'Unsupported PCKType: %s', ModelOpts.PCKType);
        end
    end
end

function scope = localResolveParallelScope(ModelOpts, total)
    if ~isfield(ModelOpts, 'Parallel') || ~ModelOpts.Parallel
        scope = 'serial';
        return;
    end

    model_type = lower(strtrim(ModelOpts.ModelType));
    if strcmp(model_type, 'apc')
        scope = localPickScope(total > 1, 'outer', 'serial');
        return;
    end

    pck_type = lower(strtrim(ModelOpts.PCKType));
    if strcmp(pck_type, 'sequential')
        scope = localPickScope(total > 1, 'outer', 'serial');
    else
        scope = localPickScope(total > 1, 'outer', 'trend');
    end
end

function out = localPickScope(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function label = localParallelScopeLabel(scope)
    switch lower(scope)
        case 'serial'
            label = 'Serial';
        case 'trend'
            label = 'Trend-Parallel';
        case 'outer'
            label = 'Parallel';
        otherwise
            label = scope;
    end
end

function localPrintTrainHeader(scope, total)
    fprintf('\n%s\n', localTrainDivider('BEGIN'));
    fprintf('[aPC_Train] (%s) Evaluating %d configuration(s)...\n', ...
        localParallelScopeLabel(scope), total);
end

function localPrintTrainFooter()
    fprintf('\n%s\n\n', localTrainDivider('END'));
end

function line = localTrainDivider(label)
    width = localTrainLogWidth() - strlength('[aPC_Train] ');
    core = sprintf(' aPC TRAIN %s ', upper(label));
    pad_total = max(0, width - strlength(core));
    left = floor(pad_total / 2);
    right = pad_total - left;
    line = ['[aPC_Train] ' repmat('=', 1, left) core repmat('=', 1, right)];
end

function line = localFormatTrainProgress(bar_str, pct, cnt, total)
    width = localTrainLogWidth();
    raw = sprintf('[aPC_Train] [%s] %3d%%  (%d/%d)', bar_str, round(pct * 100), cnt, total);
    line = sprintf(['%-' num2str(width) 's'], raw);
end

function width = localTrainLogWidth()
    width = 84;
end
