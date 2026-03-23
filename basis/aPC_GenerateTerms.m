function PCE_terms = aPC_GenerateTerms(ModelOpts)
% aPC_GenerateTerms (FAST, IDENTICAL ORDER)
% ------------------------------------------------------------
% 目标：在不改变“计算结果（集合+顺序）”的前提下加速生成多指数索引
% 与原始递归 generate_combinations 输出顺序严格一致。
%
% ModelOpts.Order    : 最大总阶 p
% ModelOpts.qnorm    : q-norm（空或>=1 等价于不截断）
% ModelOpts.num_vars : 维数 d
%
% 输出：
%   PCE_terms : double 矩阵，与原实现的类型一致（默认 double）
% ------------------------------------------------------------

    p = ModelOpts.Order;
    d = ModelOpts.num_vars;

    q = 1;
    if isfield(ModelOpts,'qnorm') && ~isempty(ModelOpts.qnorm)
        q = ModelOpts.qnorm;
    end
    if q >= 1, q = 1; end  % 与你原逻辑一致：q>=1 等价于不做 q-norm 截断

    % ---------- optional cache (不会改变结果，只是复用) ----------
    persistent CACHE
    if isempty(CACHE)
        CACHE = containers.Map('KeyType','char','ValueType','any');
    end
    key = sprintf('d%d_p%d_q%.16g', d, p, q);
    if isKey(CACHE, key)
        PCE_terms = CACHE(key);
        return
    end

    % ---------- 预分配：上界就是 total-degree ≤ p 的项数 ----------
    % 对你这个“总阶约束”的递归生成来说，这就是精确项数（q=1 时）
    % q<1 时会更少，但我们先按上界分配，再裁剪，稳定且快。
    maxN = nchoosek(d + p, p);

    % 用 uint8 存指数（p=3 时最省内存），最后转 double 保持外部一致性
    T = zeros(maxN, d, 'uint8');

    % 递归填充（严格复现你原来的生成顺序）
    row = uint32(0);
    alpha = zeros(1, d, 'uint8');

    if q < 1
        fill_with_q(1, p, 0.0);   % qsum = sum(alpha.^q) 的累积
    else
        fill_no_q(1, p);
    end

    % 裁剪 + 转 double（保持与原始输出类型一致）
    PCE_terms = double(T(1:double(row), :));

    CACHE(key) = PCE_terms;

    % ======= nested functions: 原地写入 T，避免复制 =======

    function fill_no_q(j, rem)
        % 与原 generate_combinations 完全同序：for i=0:rem
        if j == d
            for i = 0:rem
                alpha(j) = uint8(i);
                row = row + 1;
                T(row, :) = alpha;
            end
        else
            for i = 0:rem
                alpha(j) = uint8(i);
                fill_no_q(j+1, rem - i);
            end
        end
    end

    function fill_with_q(j, rem, qsum)
        % q-norm 截断的“等价实现”
        % 原代码：生成完每个 alpha 后算 (sum(alpha.^q))^(1/q) <= p
        % 这里：到叶子再做同样判定；同时加一个安全剪枝：
        %       如果当前部分 qnorm 已经 > p，则任何延伸都必失败，直接 continue
        if j == d
            for i = 0:rem
                alpha(j) = uint8(i);
                qsum2 = qsum + double(i)^q;

                % 与原 apply_qnorm_truncation 的判定一致（只差浮点误差保护）
                if (qsum2)^(1/q) <= (p + 1e-12)
                    row = row + 1;
                    T(row, :) = alpha;
                end
            end
        else
            for i = 0:rem
                alpha(j) = uint8(i);
                qsum2 = qsum + double(i)^q;

                % 安全剪枝：qsum 已经超限，则任何后续非负指数只会更大
                if (qsum2)^(1/q) > (p + 1e-12)
                    continue
                end

                fill_with_q(j+1, rem - i, qsum2);
            end
        end
    end
end
