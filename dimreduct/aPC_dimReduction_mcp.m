function [X_red, ReduceModel] = aPC_dimReduction_mcp(U, Y, ModelOpts)
% -------------------------------------------------------------------------
% aPC_dimReduction_mcp — Moment-Constrained Projection
%
%   降维矩阵 W 通过求解 Stiefel(d,r) 上的优化问题得到：
%
%       minimize   L(W) = α_cov||Cov(Z)-I||_F^2
%                    + Σ_{k=1}^K w_k Σ_j (E[Z_j^k] - E[U^k])^2
%                    + λ_y L_supervised
%
%   其中 Z = U_w * W，U_w 为白化后的输入。
%
%   使用 Manopt: rlbfgs / trustregions（比手写 Adam 更稳健）
% -------------------------------------------------------------------------

%% ===== 读取维度 =====
[N, d] = size(U);
r = ModelOpts.red_dim;

%% ===== 参数读取 =====
opts = struct();
if isfield(ModelOpts,'mcp_opts')
    opts = ModelOpts.mcp_opts;
end

opts = setdef(opts,'maxIter',800);
opts = setdef(opts,'whiten',1);
opts = setdef(opts,'alpha_cov',1.0);

opts = setdef(opts,'max_moment_order',4);
opts = setdef(opts,'moment_base',1.0);
opts = setdef(opts,'moment_decay',0.6);

opts = setdef(opts,'lambda_y',0.1);
opts = setdef(opts,'supervision_mode','corr');

opts = setdef(opts,'manopt_solver','rlbfgs');     % 'rlbfgs' 或 'trustregions'

K = opts.max_moment_order;

%% ===== 监督项准备 =====
useY = false;
if opts.lambda_y > 0 && ~isempty(Y)
    Y = Y(:);
    if strcmpi(opts.supervision_mode,'corr')
        Yc = Y - mean(Y);
    end
    useY = true;
end

%% ===== 原空间 K 阶矩 =====
U_flat = U(:); %这个和下面白化空间计算mk的部分是可替换的，测试来看没什么区别
momU = zeros(1,K);
for k = 1:K
    momU(k) = mean(U_flat.^k);
end

%% ===== Whitening =====
if opts.whiten
    C = cov(U);
    [V, D] = eig(C);
    Dinv = diag(1./sqrt(max(diag(D),0) + 1e-12));
    Wwhite = V * Dinv * V';
else
    Wwhite = eye(d);
end
U_w = U * Wwhite;

%% ===== 在白化空间计算参考矩 mk（momU）=====
% U_flat = U_w(:);                % <-- 关键：用 U_w 而不是 U
% momU   = zeros(1,K);
% for k = 1:K
%     momU(k) = mean(U_flat.^k);
% end

%% ===== 初始化 W =====
[Vp,~] = eigs(cov(U_w), r, 'largestreal');
W0 = orth(Vp + 0.01*randn(d,r));

%% ===== 构造 Manopt 问题 =====
M = stiefelfactory(d,r, true);

problem.M = M;

problem.cost = @(W) cost_fun(W, U_w, r, K, opts, momU, ...
                            useY, Y, exist('Yc','var'), N);

problem.egrad = @(W) grad_fun(W, U_w, r, K, opts, momU, ...
                             useY, Y, exist('Yc','var'), N);

%% ===== 求解 =====
options.maxiter = opts.maxIter;
options.verbosity = 1;
% options.tolgradnorm = 1e-10;

switch lower(opts.manopt_solver)
    case 'rlbfgs'
        [W, cost, info, options] = rlbfgs(problem, W0, options);
    case 'trustregions'
        [W, cost, info, options] = trustregions(problem, W0, options);
    otherwise
        warning('未知 solver，使用 rlbfgs');
        [W, cost, info, options] = rlbfgs(problem, W0, options);
end

%% ===== 输出 =====
X_red = U * (Wwhite * W);

ReduceModel.method = 'mcp';
ReduceModel.params.W = W;
ReduceModel.params.Wwhite = Wwhite;
ReduceModel.transform = @(U_new)(U_new * (Wwhite * W));

ReduceModel.info.info = info;
ReduceModel.info.opts = opts;

end

%% ========================================================================
%%                             成本函数
%% ========================================================================
function L = cost_fun(W, U_w, r, K, opts, momU, useY, Y, hasYc, N)

Z = U_w * W;

%% 二阶
Cz = cov(Z);
L_cov = norm(Cz - eye(r),'fro')^2;

%% 边缘矩
L_mom = 0;
for k = 1:K
    wk  = opts.moment_base * opts.moment_decay^(k-1);
    mk  = mean(Z.^k,1);
    diffk = mk - momU(k);
    L_mom = L_mom + wk * sum(diffk.^2);
end

%% 监督
Ly = 0;
if useY
    switch opts.supervision_mode
        case 'pls'
            Yc = Y - mean(Y);
            t = (Z' * Yc) / N;     % Cov(Z,Y)
            Ly = -(t' * t);        % ||Cov(Z,Y)||^2
        case 'corr'
            Yc = Y - mean(Y);
            t = (Z' * Yc) / N;
            Ly = -(t'*t);
    end
end

L = opts.alpha_cov*L_cov + L_mom + opts.lambda_y*Ly;

end

%% ========================================================================
%%                          欧氏梯度
%% ========================================================================
function G = grad_fun(W, U_w, r, K, opts, momU, useY, Y, hasYc, N)

Z = U_w * W;

%% dL/dZ 二阶
Cz = cov(Z);
dZ = (4*opts.alpha_cov/N) * (Z * (Cz - eye(r)));

%% dL/dZ 边缘矩
for k = 1:K
    wk  = opts.moment_base * opts.moment_decay^(k-1);
    mk  = mean(Z.^k,1);
    diffk = mk - momU(k);
    grad_factor = (2*wk*k/N) * diffk;
    dZ = dZ + (Z.^(k-1)) .* grad_factor;
end

%% dL/dZ 监督
if useY
    switch opts.supervision_mode
        case 'pls'
            Yc = Y - mean(Y);
            t = (Z' * Yc) / N;                 % Cov(Z,Y)
            dZ = dZ - opts.lambda_y * (2/N) * (Yc * t');
        case 'corr'
            Yc = Y - mean(Y);
            t = (Z' * Yc) / N;
            dZ = dZ - opts.lambda_y * (2/N) * (Yc * t');
    end
end

%% 链式到 W
G = U_w' * dZ;

end

%% ========================================================================
function opts = setdef(opts, field, default)
if ~isfield(opts,field) || isempty(opts.(field))
    opts.(field) = default;
end
end
