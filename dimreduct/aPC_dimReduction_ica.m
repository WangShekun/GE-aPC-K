function [X_red, ReduceModel] = aPC_dimReduction_ica(U, ModelOpts)
% aPC_dimReduction_ica
% ---------------------------------------------------------
% ICA (FastICA-style fixed-point) 降维适配器（unsupervised）
%
% 参考主流开源实现（FastICA 2.5 及其在诸多 GitHub 工程中的外部依赖形态）
% 的“去均值-白化-固定点迭代(对称/逐个)-稳定化/步长”标准流程。
%
% 输入：
%   U         — 标准化输入 (N×M)
%   ModelOpts — 控制结构体（只读下面字段）
%
% 必需字段：
%   ModelOpts.red_dim            : r（整数>0，目标独立成分数/降维维数）
%
% 可选字段（FastICA-like，推荐使用 ModelOpts.ica_opts.*）：
%   ModelOpts.ica_opts.approach       : 'symm'(默认) | 'defl'
%   ModelOpts.ica_opts.fun            : 'tanh'(默认) | 'gauss' | 'pow3' | 'cube'
%   ModelOpts.ica_opts.epsilon        : 收敛阈值(默认 1e-4)
%   ModelOpts.ica_opts.maxIter        : 最大迭代(默认 200)
%   ModelOpts.ica_opts.mu             : 步长/阻尼(默认 1.0, (0,1] )
%   ModelOpts.ica_opts.stabilization  : 'off'(默认) | 'on'
%   ModelOpts.ica_opts.a1             : tanh 的 a1(默认 1.0)
%   ModelOpts.ica_opts.a2             : gauss 的 a2(默认 1.0)
%   ModelOpts.ica_opts.seed           : 随机种子(默认空)
%   ModelOpts.ica_opts.whiten_solver  : 'svd'(默认) | 'eig'
%   ModelOpts.ica_opts.verbose        : 0(默认) | 1
%
% 输出：
%   X_red       — (N×r) 训练降维空间（估计的独立成分）
%   ReduceModel — 含 transform：U_new -> Z_new
% ---------------------------------------------------------

    %% ====== 0) red_dim 检查 ======
    if ~isfield(ModelOpts,'red_dim') || isempty(ModelOpts.red_dim)
        error('ICA requires ModelOpts.red_dim.');
    end
    r_req = ModelOpts.red_dim;
    if numel(r_req) ~= 1 || r_req <= 0 || floor(r_req) ~= r_req
        error('ICA red_dim must be a positive integer scalar.');
    end

    [N, M] = size(U);
    eps0 = 1e-12;

    %% ====== 1) 读取参数（FastICA-like） ======
    opts_in = struct();
    if isfield(ModelOpts, 'ica_opts') && ~isempty(ModelOpts.ica_opts)
        opts_in = ModelOpts.ica_opts;
    end

    opts = struct();
    opts.approach      = local_getopt(opts_in, 'approach',      'symm');
    opts.fun           = local_getopt(opts_in, 'fun',           'tanh');
    opts.epsilon       = local_getopt(opts_in, 'epsilon',       1e-4);
    opts.maxIter       = local_getopt(opts_in, 'maxIter',       200);
    opts.mu            = local_getopt(opts_in, 'mu',            1.0);
    opts.stabilization = local_getopt(opts_in, 'stabilization', 'off');
    opts.a1            = local_getopt(opts_in, 'a1',            1.0);
    opts.a2            = local_getopt(opts_in, 'a2',            1.0);
    opts.seed          = local_getopt(opts_in, 'seed',          []);
    opts.whiten_solver = local_getopt(opts_in, 'whiten_solver', 'svd');
    opts.verbose       = local_getopt(opts_in, 'verbose',       0);

    opts.mu = max(min(opts.mu, 1.0), eps0); % clamp to (0,1]

    %% ====== 2) 去均值（FastICA 常用 remmean） ======
    muX = mean(U, 1);
    Xc  = bsxfun(@minus, U, muX);

    %% ====== 3) PCA 白化（FastICA 常用 pcamat: whitening + dewhitening） ======
    % 这里把 PCA 截断维数 = r（与你的“降维维数=输出维数”保持一致）
    switch lower(opts.whiten_solver)
        case 'svd'
            [~, S, V] = svd(Xc, 'econ');              % Xc = U_s*S*V'
            s = diag(S);
            rk = numel(s);
            r  = min([r_req, rk, M]);

            % cov eigenvalues: d = s.^2/(N-1)
            scale = sqrt(max(N-1,1));
            E = V(:,1:r);
            d = (s(1:r).^2) / max(N-1,1);

            % whitening right-multiplier: Xw = Xc * (E * diag(1/sqrt(d)))
            Pwhite = E * diag(1 ./ sqrt(d + eps0));
            Xw = Xc * Pwhite;                         % N×r

        case 'eig'
            C = (Xc' * Xc) / max(N-1,1);
            [Eall, Dall] = eig((C + C')/2);
            dAll = diag(Dall);
            [dAll, idx] = sort(dAll, 'descend');
            Eall = Eall(:, idx);

            rk = sum(dAll > eps0);
            r  = min([r_req, rk, M]);

            E = Eall(:,1:r);
            d = dAll(1:r);

            Pwhite = E * diag(1 ./ sqrt(d + eps0));
            Xw = Xc * Pwhite;

        otherwise
            error('Unknown ica_whiten_solver: %s', opts.whiten_solver);
    end

    %% ====== 4) FastICA fixed-point（symm/defl） ======
    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    X = Xw'; % r×N（FastICA 传统是“每行一个信号”）

    if strcmpi(opts.approach,'defl')
        [W, info] = local_fpica_defl(X, opts, eps0);
    else
        [W, info] = local_fpica_symm(X, opts, eps0);
    end

    % sources: S = (W*X)'  -> N×r
    X_red = (W * X)';

    %% ====== 5) ReduceModel 封装（跟你 KPCA 风格一致） ======
    params = struct();
    params.muX    = muX;
    params.Pwhite = Pwhite;           % Xw = (U-muX) * Pwhite
    params.W      = W;                % unmixing in whitened space
    params.Worig  = W * Pwhite';      % S = (Worig*(U-muX)')'
    params.A      = pinv(params.Worig);

    ReduceModel = struct();
    ReduceModel.method = 'ica';
    ReduceModel.params = params;

    ReduceModel.transform = @(U_new) local_ica_predict(U_new, params);

    ReduceModel.info = info;

    if opts.verbose
        fprintf('[ICA] approach=%s, g=%s, r=%d, it=%d, final=%.3e\n', ...
            opts.approach, opts.fun, size(W,1), info.nIter, info.convHist(end));
    end
end

%% ============================================================
% Prediction：U_new -> Z_new
%% ============================================================
function Z = local_ica_predict(U_new, params)
    Xc_new = bsxfun(@minus, U_new, params.muX);
    Xw_new = Xc_new * params.Pwhite;    % N×r
    Z      = Xw_new * params.W';        % N×r
end

%% ============================================================
% Symmetric (parallel) fixed-point ICA
%% ============================================================
function [W, info] = local_fpica_symm(X, opts, eps0)
    [r, N] = size(X);

    W = randn(r, r);
    W = local_sym_decor(W, eps0);

    convHist = zeros(opts.maxIter,1);
    mu = opts.mu;
    prevLim = inf;

    for it = 1:opts.maxIter
        Wold = W;

        WX = W * X; % r×N
        [gWX, gpWX] = local_g(WX, opts);

        Wnew = (gWX * X') / N - diag(mean(gpWX,2)) * W;

        % damping / stepsize (FastICA-like "mu")
        W = (1-mu) * Wold + mu * Wnew;
        W = local_sym_decor(W, eps0);

        lim = max(abs(abs(diag(W * Wold')) - 1));
        convHist(it) = lim;

        % stabilization：如果发散趋势明显，临时减小 mu
        if strcmpi(opts.stabilization,'on')
            if lim > prevLim * 1.05
                mu = max(mu * 0.5, 0.1);   % 保底别太小
                W  = Wold;                 % 回退一步更稳
            else
                mu = min(mu * 1.05, opts.mu);
            end
            prevLim = lim;
        end

        if lim < opts.epsilon
            convHist = convHist(1:it);
            info = struct('nIter', it, 'convHist', convHist, 'mu_final', mu);
            return;
        end
    end

    info = struct('nIter', opts.maxIter, 'convHist', convHist, 'mu_final', mu);
end

%% ============================================================
% Deflationary fixed-point ICA
%% ============================================================
function [W, info] = local_fpica_defl(X, opts, eps0)
    [r, N] = size(X);

    W = zeros(r, r);
    convHist = zeros(opts.maxIter,1);
    mu0 = opts.mu;

    maxItUsed = 0;

    for p = 1:r
        w = randn(r,1);
        w = w / (norm(w) + eps0);

        mu = mu0;
        prevLim = inf;

        for it = 1:opts.maxIter
            wold = w;

            wx = w' * X; % 1×N
            [gwx, gpwx] = local_g(wx, opts);

            wnew = (X * gwx') / N - mean(gpwx) * w;

            % deflation orthogonalization
            if p > 1
                Wp = W(1:p-1,:)';   % r×(p-1)
                wnew = wnew - Wp * (Wp' * wnew);
            end

            % damping
            w = (1-mu) * wold + mu * wnew;
            w = w / (norm(w) + eps0);

            lim = abs(abs(w' * wold) - 1);
            convHist(it) = max(convHist(it), lim);

            if strcmpi(opts.stabilization,'on')
                if lim > prevLim * 1.05
                    mu = max(mu * 0.5, 0.1);
                    w  = wold;
                else
                    mu = min(mu * 1.05, mu0);
                end
                prevLim = lim;
            end

            if lim < opts.epsilon
                maxItUsed = max(maxItUsed, it);
                break;
            end
        end

        W(p,:) = w';
    end

    convHist = convHist(1:max(maxItUsed,1));
    info = struct('nIter', max(maxItUsed,1), 'convHist', convHist, 'mu_final', mu0);
end

%% ============================================================
% Nonlinearity g(u), g'(u)  (FastICA 常见：tanh/gauss/pow3)
%% ============================================================
function [gU, gpU] = local_g(U, opts)
    switch lower(opts.fun)
        case {'tanh','logcosh'}
            a1 = opts.a1;
            gU  = tanh(a1 * U);
            gpU = a1 * (1 - gU.^2);

        case {'gauss','gaussian'}
            a2 = opts.a2;
            gU  = U .* exp(-0.5 * a2 * U.^2);
            gpU = (1 - a2 * U.^2) .* exp(-0.5 * a2 * U.^2);

        case {'pow3','cube'}
            gU  = U.^3;
            gpU = 3 * U.^2;

        otherwise
            error('Unknown ica_fun: %s (use tanh/gauss/pow3)', opts.fun);
    end
end

%% ============================================================
% Symmetric decorrelation: W <- (W W')^{-1/2} W
%% ============================================================
function W = local_sym_decor(W, eps0)
    C = W*W';
    C = (C + C')/2;
    [E, D] = eig(C);
    d = diag(D);
    W = (E * diag(1 ./ sqrt(d + eps0)) * E') * W;
end

function v = local_getopt(S, field, defaultVal)
    if isfield(S, field) && ~isempty(S.(field))
        v = S.(field);
    else
        v = defaultVal;
    end
end
