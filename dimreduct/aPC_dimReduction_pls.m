function [X_red, ReduceModel] = aPC_dimReduction_pls(U, Y, ncomp)
% aPC_dimReduction_pls
% -------------------------------------------------------
% PLS 降维模块（与主函数解耦，可单独调用）
%
% 输入：
%   U      — 标准化输入 (N×M)
%   Y      — 输出 (N×1)
%   ncomp  — 目标维度
%
% -------------------------------------------------------

    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = ...
        plsregress(U, Y, ncomp, 'Intercept', false);

    params.W  = stats.W;   % 降维矩阵
    ReduceModel.method = 'pls';
    ReduceModel.params = params;

    % transform 回调：用于预测
    ReduceModel.transform = @(U_new)( U_new * params.W );

    % 完整信息
    info.XL     = XL;
    info.YL     = YL;
    info.XS     = XS;
    info.YS     = YS;
    info.BETA   = BETA;
    info.PCTVAR = PCTVAR;
    info.MSE    = MSE;
    info.stats  = stats;

    ReduceModel.info = info;

    % 降维后的训练数据（XS）
    X_red = XS;
end
