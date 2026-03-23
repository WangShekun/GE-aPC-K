function [X_red, ReduceModel] = aPC_dimReduction_pca(U, ncomp)
% aPC_dimReduction_pca
% -------------------------------------------------------
% PCA 降维模块（与主函数解耦）
%
% 输入：
%   U      — 标准化输入 (N×M)
%   ncomp  — 目标维度
% -------------------------------------------------------

    [coeff, score, latent] = pca(U);

    params.V = coeff(:,1:ncomp);

    ReduceModel.method = 'pca';
    ReduceModel.params = params;

    ReduceModel.transform = @(U_new)( U_new * params.V );

    info.coeff  = coeff;
    info.score  = score;
    info.latent = latent;

    ReduceModel.info = info;

    X_red = score(:, 1:ncomp);
end
