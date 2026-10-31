function y_pred = aPC_Predict(X_new, results, ModelOpts, ReduceModel)
% aPC_Predict：支持任意降维方法（PLS, PCA, ICA…）

% ====== 1) 标准化，使用训练时的 mu & sigma ======
U_new = results.map_U.transform(X_new);

% ====== 2) 降维（统一接口） ======
if ~isempty(ReduceModel)
    U_new = ReduceModel.transform(U_new);
end

M = size(U_new, 2);   % 降维后的维数
N = size(U_new, 1);

% ====== 3) 多项式基构造 ======
poly_coeffs = results.poly_coeffs;
Ind = results.Indices;
c = results.coefficients(:);

Pmax = max(Ind,[],'all');
univ_vals_new = zeros(N, M, Pmax+1);

for j = 1:M
    coeffs_j = poly_coeffs{j};
    u_j = U_new(:,j);
    for k = 0:Pmax
        univ_vals_new(:,j,k+1) = polyval(coeffs_j{k+1}, u_j);
    end
end

% ====== 4) 多维 Ψ 构造 ======
Psi_new = aPC_PCE_create_Psi(Ind, univ_vals_new);

% ====== 5) 最终预测 ======
if isfield(results, 'ModelType') && ~strcmpi(results.ModelType, 'apc')
    if strcmpi(results.ModelType, 'geapck_lowrank')
        y_pred = aPC_PCKPredictLowRank(Psi_new, U_new, results.PCKModel);
    else
        y_pred = aPC_PCKPredict(Psi_new, U_new, results.PCKModel);
    end
else
    y_pred = Psi_new * c;
end
end
