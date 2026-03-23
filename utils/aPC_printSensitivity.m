function aPC_printSensitivity(SA, ModelOpts)
% aPC_printSensitivity
% -------------------------------------------------------
% 打印基于 aPC 的 Sobol 灵敏度分析结果（1 阶、2 阶、总效应）
%
% 输入：
%   SA.S_first   - 1×M 一阶 Sobol 指数
%   SA.S_total   - 1×M 总 Sobol 指数
%   SA.S_second  - M×M 二阶 Sobol 指数组
%   ModelOpts.num_vars - 输入变量维度
% -------------------------------------------------------

M = ModelOpts.num_vars;

line  = repmat('-',1,70);
dline = repmat('=',1,70);

%% ===== 打印头部 =====
fprintf('\n%s\n', dline);
fprintf('                 Sobol Sensitivity Analysis Report\n');
fprintf('%s\n', line);

names = aPC_varNames(M);

%% --------- 总效应 Sobol ---------
fprintf('>>> Total Sobol'' Indices (Order ≤ 2)\n');
fprintf('%s\n', line);

printHeader(names);
fprintf('%s\n', line);
printRow(SA.S_total);
fprintf('%s\n\n', line);


%% --------- 一阶 Sobol ---------
fprintf('>>> First-order Sobol'' Indices\n');
fprintf('%s\n', line);

printHeader(names);
fprintf('%s\n', line);
printRow(SA.S_first);
fprintf('%s\n\n', line);


%% --------- 二阶 Sobol ---------
fprintf('>>> Second-order Sobol'' Indices\n');
fprintf('%s\n', line);

% 生成二阶变量名 XiXj
second_names = {};
second_vals  = [];
for i = 1:M
    for j = i+1:M
        second_names{end+1} = sprintf('%s%s', names{i}, names{j});
        second_vals(end+1)  = SA.S_second(i,j);
    end
end

printHeader(second_names);
fprintf('%s\n', line);
printRow(second_vals);
fprintf('%s\n', line);


%% ===== 打印尾部 =====
fprintf('%s\n', dline);


%% ------------------------------------
% 辅助打印函数
% ------------------------------------
function printHeader(names)
    for i = 1:numel(names)
        fprintf('%-12s', names{i});
    end
    fprintf('\n');
end

function printRow(vals)
    for i = 1:numel(vals)
        fprintf('%-12.6f', vals(i));
    end
    fprintf('\n');
end

end

%% -----------------------
% 变量名生成
% -----------------------
function names = aPC_varNames(M)
names = arrayfun(@(i) sprintf('X%d',i), 1:M, 'UniformOutput', false);
end
