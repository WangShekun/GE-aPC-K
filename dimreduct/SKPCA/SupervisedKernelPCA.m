classdef SupervisedKernelPCA < handle
    % Supervised Kernel PCA (HSIC-based supervised PCA)
    % ------------------------------------------------
    % 完全模仿 KernelPCA 的结构，但加入监督核 Ky。
    % 用于回归型任务（连续 Y），不包含 delta。
    %
    % 支持：
    %   - kernelFunc: Kernel for X
    %   - kernelY   : Kernel for Y (linear / gaussian / polynomial)
    %   - numComponents: 指定成分个数或解释率
    %
    % 输出：
    %   .score        投影结果
    %   .coefficient  特征向量
    %   .lambda       特征值
    %
    % 打印：
    %   display = 'on' 时模仿 KernelPCA 的输出风格
    %

    properties
        data               % X
        Y                  % 输出（连续变量）
        numComponents      % 主成分数 / 解释率阈值
        kernelFunc         % 输入核
        kernelY            % 输出核

        lambda             % 特征值
        coefficient        % 特征向量
        score              % 投影

        temporary          % 保存 Kx/H/Kxc
        runningTime
        display = 'on'
    end

    properties (Dependent)
        numSamples
    end


    methods
        function obj = SupervisedKernelPCA(parameter)
            names = fieldnames(parameter);
            for i = 1:numel(names)
                obj.(names{i}) = parameter.(names{i});
            end
        end


        % ============================
        %            TRAIN
        % ============================
        function obj = train(obj, X, Y)
            tStart = tic;

            obj.data = X;
            obj.Y    = Y;
            N = obj.numSamples;

            %% ---- 1) Kx（输入核） ----
            Kx = obj.kernelFunc.computeMatrix(X, X);

            H  = eye(N) - ones(N)/N;
            Kxc = H * Kx * H;

            %% ---- 2) Ky（监督核） ----
            Ky  = obj.kernelY.computeMatrix(Y, Y);
            Kyc = H * Ky * H;

            %% ---- 3) HSIC-like 矩阵 ----
            % M = Kxc * Kyc * Kxc
            M = Kxc * Kyc * Kxc;

            %% ---- 4) 对称化 + 特征分解 ----
            [V, D] = eig((M + M')/2);
            lambda = diag(D);

            % 降序排序
            [lambda, idx] = sort(lambda, 'descend');
            V = V(:, idx);

            %% ---- 5) 选择主成分 ----
            if obj.numComponents <= 1  % 用解释率
                cum = cumsum(lambda) / sum(lambda);
                obj.numComponents = find(cum >= obj.numComponents, 1, 'first');
            end

            r = obj.numComponents;
            obj.lambda = lambda(1:r);
            obj.coefficient = V(:, 1:r);

            %% ---- 6) 得到投影 ----
            obj.score = Kxc * obj.coefficient;

            %% ---- 7) 保存中间量 ----
            obj.temporary.Kx  = Kx;
            obj.temporary.Kxc = Kxc;
            obj.temporary.H   = H;

            obj.runningTime = toc(tStart);

            %% ---- 打印 ----
            if strcmp(obj.display,'on')
                SupervisedKernelPCAOption.displayTrain(obj);
            end
        end


        % ============================
        %            TEST
        % ============================
        function result = test(obj, Xnew)
            tStart = tic;

            Ntr = obj.numSamples;
            Nte = size(Xnew,1);

            %% ---- test-train kernel ----
            Kt = obj.kernelFunc.computeMatrix(Xnew, obj.data);

            %% ---- 中心化（模仿 KPCA） ----
            H   = obj.temporary.H;
            Kx  = obj.temporary.Kx;
            unit_te = ones(Nte,Ntr)/Ntr;

            Kt_c = Kt ...
                - unit_te*Kx ...
                - Kt*(ones(Ntr)/Ntr) ...
                + unit_te*Kx*(ones(Ntr)/Ntr);

            %% ---- 投影 ----
            result.score = Kt_c * obj.coefficient;

            result.runningTime = toc(tStart);

            %% ---- 打印 ----
            if strcmp(obj.display,'on')
                SupervisedKernelPCAOption.displayTest(result);
            end
        end


        % ============================
        %       GET number of samples
        % ============================
        function N = get.numSamples(obj)
            N = size(obj.data,1);
        end
    end
end
