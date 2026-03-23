function out = aPC_evaluateCandidate_score(data, ModelOpts)
% 并行候选评估：返回 out.score（用于搜索）

candidate = aPC_TrainCandidate(data, data.map_U, ModelOpts);
out.score = candidate.score;
end
