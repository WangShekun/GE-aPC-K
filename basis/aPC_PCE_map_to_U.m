function [data, map_U] = aPC_PCE_map_to_U(X, data)
% 标准化映射，返回 transform 函数句柄形式的 map_U

    % ---- 1. 计算均值、标准差 ----
    [Z, mu, sigma] = zscore(X);

    % ---- 2. 去除零方差维度 ----
    nonConstIdx = sigma > 0;

    Z     = Z(:, nonConstIdx);
    mu    = mu(nonConstIdx);
    sigma = sigma(nonConstIdx);

    % ---- 3. 保存到 data ----
    data.U = Z;

    % ---- 4. 构造 transform 函数句柄 ----
    % 注意：由于维度筛选，必须先映射，再筛选
    map_U.transform = @(Xnew) ((Xnew - mu) ./ sigma);

    % 若需要反标准化，可用 untransform
    map_U.untransform = @(Unew) (Unew .* sigma + mu);

    % ---- 5. 附加信息，用于调试或记录 ----
    map_U.mu          = mu;
    map_U.sigma       = sigma;
    map_U.nonConstIdx = nonConstIdx;
end
