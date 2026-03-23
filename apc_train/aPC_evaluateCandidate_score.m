function out = aPC_evaluateCandidate_score(data, ModelOpts)
% 并行候选评估：返回 out.score（用于搜索）
% 评分：仅使用 LOO（训练/搜索阶段不使用验证集）

% —— 拷贝一份，避免修改外部结构 —— 
data_loc      = data;
Modelopts_loc = ModelOpts;

% —— 降维：如未指定 red_dim 则跳过 —— 
if isfield(Modelopts_loc,'red_dim') && ~isempty(Modelopts_loc.red_dim)
    [data_loc, Modelopts_loc, ReduceModel] = aPC_dimReduction(data_loc, Modelopts_loc);
else
    ReduceModel = [];
    if ~isfield(Modelopts_loc,'num_vars') || isempty(Modelopts_loc.num_vars)
        Modelopts_loc.num_vars = size(data_loc.U,2);
    end
end

% —— 多索引 & 基函数 —— 
Modelopts_loc.Indices = aPC_GenerateTerms(Modelopts_loc);
[Psi_tmp, ~, ~] = aPC_PCE_build_Psi(Modelopts_loc, data_loc);

% —— 回归 —— 
res_tmp = aPC_RegressionMethods(Psi_tmp, data_loc.Y_train, Modelopts_loc.RegressionMethod);

% —— 评分（仅 LOO）——
score = res_tmp.LOO;

out.score = score;
end
