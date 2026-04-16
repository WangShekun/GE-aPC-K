# aPC MATLAB Framework

这份文档只解释核心程序，不写 `tmp/` 下的临时实验脚本。结构按“文件夹 / 子文件”展开，方便直接对应代码阅读。

## 运行入口

- 先运行 `core/aPC.m` 初始化路径。
- 若使用 benchmark 输入或 UQLab 基线，再运行 `uqlab`。
- 单次实验入口：`main.m`
- 对比实验入口：`main_compare.m`

常见主程序设置字段示例：

```matlab
ModelOpts.Order = 4;                        % 多项式总阶数
ModelOpts.qnorm = 0.8;                     % q-norm 截断参数
ModelOpts.aPC_Method = 'stieltjes';        % 一维 aPC 构造方法
ModelOpts.RegressionMethod = 'lars';       % 回归器 / trend 构造器常用设置
ModelOpts.TrendBuilder = 'lars';           % PCK 趋势路径构造器
ModelOpts.ModelType = 'apck';              % 模型类型：apc / apck / geapck / geapck_lowrank
ModelOpts.PCKType = 'optimal';             % PCK 策略：sequential / optimal
ModelOpts.GradientMode = 'provided';       % GE 梯度来源：provided / finite-difference / auto
ModelOpts.Parallel = 0;                    % 是否允许并行
ModelOpts.ReductionMethod = 'pls';         % 降维方法
ModelOpts.red_dim = 4;                     % 降维后的目标维数
ModelOpts.KrigingOpts.corr = 'gaussian';   % Kriging 相关函数类型
ModelOpts.KrigingOpts.theta0 = 1.0;        % 长度尺度初值
ModelOpts.KrigingOpts.theta_lb = 1e-3;     % 长度尺度下界
ModelOpts.KrigingOpts.theta_ub = 1e2;      % 长度尺度上界
ModelOpts.KrigingOpts.nugget = 1e-8;       % 数值稳定项
ModelOpts.KrigingOpts.optimize_theta = 1;  % 是否优化 theta
ModelOpts.KrigingOpts.trend_range = [0.1 0.8]; % PCK trend 搜索范围
ModelOpts.GELowRankOpts.rank = 3;          % GE low-rank 保留的梯度模态数
ModelOpts.FDOpts.type = 'central';         % 数值梯度类型
ModelOpts.FDOpts.step = 1e-6;              % 数值梯度相对步长
```

训练数据常见字段示例：

```matlab
data.X_train = Xtrain;                     % 训练输入，N x M
data.Y_train = Ytrain;                     % 训练输出，N x 1
data.X_val = Xval;                         % 验证输入
data.Y_val = Yval;                         % 验证输出
data.dYdX_train = dYdX;                    % 训练梯度，N x M
data.ModelFcn = @my_model;                 % 原模型函数，用于 finite-difference
```

## 1. `core/`

### 1.1 `aPC.m`
功能：
- 初始化工程路径。
- 把 `basis/`、`apc_train/`、`coefficients/`、`predict/`、`utils/`、`dimreduct/`、`benchmark/` 等目录加入 MATLAB 路径。

调用位置：
- `main.m`
- `main_compare.m`

相关设置字段：
- 无。

## 2. `apc_train/`

### 2.1 `aPC_Train.m`
功能：
- 训练总入口。
- 遍历候选配置并选择最优模型。
- 汇总 `results`、`ModelOpts`、`data`、`ReduceModel`。

调用位置：
- `main.m`
- `main_compare.m`

相关设置字段：

```matlab
ModelOpts.Order = [4 5 6];                 % 可写搜索向量
ModelOpts.qnorm = [0.7 0.9 1.0];          % 可写搜索向量
ModelOpts.ModelType = 'apck';              % 决定走 APC 还是 PCK 家族
ModelOpts.Parallel = 1;                    % 是否允许训练阶段并行
```

### 2.2 `aPC_TrainCandidate.m`
功能：
- 训练单个 candidate。
- 调用标准化、降维、梯度准备、`Psi` 构造。
- 根据 `ModelType` 分发到 APC / APCK / GEAPCK / GEAPCK_LOWRANK 后端。

相关设置字段：

```matlab
ModelOpts.ModelType = 'geapck';            % 选择具体后端
ModelOpts.ReductionMethod = 'pls';         % 若设置，则先做降维
ModelOpts.red_dim = 4;                     % 降维目标维数
ModelOpts.GradientMode = 'provided';       % GE 路径下准备梯度的方式
```

### 2.3 `aPC_evaluateCandidate_score.m`
功能：
- candidate 评分辅助函数。
- 用于候选模型比较时统一抽取分数。

相关设置字段：
- 无独立字段，跟随 `aPC_Train.m` 的候选评分逻辑。

## 3. `basis/`

### 3.1 `aPC_PCE_map_to_U.m`
功能：
- 把原始输入 `X` 映射到标准化空间 `U`。
- 后续多项式、降维、PCK 都基于 `U` 运行。

相关设置字段：
- 无直接字段，由训练流程自动调用。

### 3.2 `aPC_GenerateTerms.m`
功能：
- 根据 `Order` 和 `qnorm` 生成多项式多指标项。

相关设置字段：

```matlab
ModelOpts.Order = 4;                       % 多项式总阶数
ModelOpts.qnorm = 0.8;                    % 控制高阶交互项稀疏程度
```

### 3.3 `aPC_PCE_build_Psi.m`
功能：
- 构造设计矩阵 `Psi`。
- 同时缓存后续求导要用的一维基信息。

相关设置字段：

```matlab
ModelOpts.aPC_Method = 'stieltjes';       % 一维基构造方法
```

### 3.4 `aPC_PCE_create_Psi.m`
功能：
- `Psi` 构造辅助函数。
- 在指定多项式项后生成对应矩阵。

### 3.5 `aPC_eval_univ_basis.m`
功能：
- 评估一维基函数值。

### 3.6 `aPC_eval_aPC_2012.m`
功能：
- 2012 版本 aPC 一维基构造。

相关设置字段：

```matlab
ModelOpts.aPC_Method = '2012';            % 使用 2012 版本一维基
```

### 3.7 `aPC_eval_aPC_2018.m`
功能：
- 2018 版本 aPC 一维基构造。

相关设置字段：

```matlab
ModelOpts.aPC_Method = '2018';            % 使用 2018 版本一维基
```

### 3.8 `aPC_eval_aPC_Stieltjes.m`
功能：
- Stieltjes 版本一维基构造。
- 当前主程序最常用的默认选择。

相关设置字段：

```matlab
ModelOpts.aPC_Method = 'stieltjes';       % 使用 Stieltjes 版本一维基
```

## 4. `coefficients/`

### 4.1 `aPC_RegressionMethods.m`
功能：
- APC 路径的回归入口。
- 根据 `RegressionMethod` 分发到 OLS / LARS / Lasso / Ridge / OMP。

相关设置字段：

```matlab
ModelOpts.ModelType = 'apc';              % 纯 APC 时走这里
ModelOpts.RegressionMethod = 'lars';      % 回归器类型
```

### 4.2 `aPC_OLS.m`
功能：
- OLS 回归。

### 4.3 `aPC_LARS.m`
功能：
- LARS 回归。
- 也是 PCK 趋势路径的重要来源。

相关设置字段：

```matlab
ModelOpts.RegressionMethod = 'lars';      % 使用 LARS 回归
ModelOpts.TrendBuilder = 'lars';          % PCK 趋势路径也基于 LARS
```

### 4.4 `aPC_Lasso.m`
功能：
- Lasso 回归。

### 4.5 `aPC_Ridge.m`
功能：
- Ridge 回归。

### 4.6 `aPC_OMP.m`
功能：
- OMP 回归。

### 4.7 `aPC_PCKBackend.m`
功能：
- APCK / GEAPCK 主后端。
- 构造趋势项矩阵 `F`。
- 在 `optimal` 模式下沿趋势路径搜索。
- 进行 Kriging 拟合、theta 优化、LOO 计算。
- `geapck` 时会进入增广系统，把梯度块一起并入训练。

相关设置字段：

```matlab
ModelOpts.ModelType = 'apck';             % APCK 主后端
ModelOpts.ModelType = 'geapck';           % GEAPCK 主后端
ModelOpts.PCKType = 'optimal';            % optimal: 搜索 trend
ModelOpts.PCKType = 'sequential';         % sequential: 固定最终 trend 支撑集
ModelOpts.TrendBuilder = 'lars';          % trend 路径构造器
ModelOpts.KrigingOpts.corr = 'gaussian';  % 相关函数类型
ModelOpts.KrigingOpts.theta0 = 1.0;       % theta 初值
ModelOpts.KrigingOpts.theta_lb = 1e-3;    % theta 下界
ModelOpts.KrigingOpts.theta_ub = 1e2;     % theta 上界
ModelOpts.KrigingOpts.nugget = 1e-8;      % 数值稳定项
ModelOpts.KrigingOpts.optimize_theta = 1; % 是否优化 theta
ModelOpts.KrigingOpts.trend_range = [0.1 0.8]; % trend 搜索区间
ModelOpts.KrigingOpts.DisplayProgress = 1; % 是否显示内部进度
ModelOpts.KrigingOpts.early_stop_patience = 3; % 连续若干 trend 无改进则提前停
```

### 4.8 `aPC_PCKBackendLowRankGE.m`
功能：
- `geapck_lowrank` 后端。
- 先在模型空间梯度上做低秩模态压缩，再构造 GE 增广系统。
- 当前使用 SVD 提取主导梯度模态。

相关设置字段：

```matlab
ModelOpts.ModelType = 'geapck_lowrank';   % 启用 GE low-rank 路径
ModelOpts.PCKType = 'optimal';            % 是否搜索 trend
ModelOpts.GELowRankOpts.rank = 3;         % 保留的梯度模态数
ModelOpts.GELowRankOpts.method = 'svd';   % 当前支持的低秩方法
```

## 5. `coefficients/pck/`

### 5.1 `pckCorrMatrix.m`
功能：
- 计算 APCK 的相关矩阵。

### 5.2 `pckCrossCovariance.m`
功能：
- 计算训练点和预测点之间的交叉协方差。

### 5.3 `pckEvalKernel.m`
功能：
- GE / GE block 路径的核评估总入口。
- 统一返回 `Kff`、`Kfg`、`Kgg` 或 block 形式的梯度块。

### 5.4 `pckBuildAugmentedSystem.m`
功能：
- 构造 full GEAPCK 的增广系统。
- 把函数值和梯度一起写进联合矩阵。

### 5.5 `kernels/pckKernelGaussian.m`
功能：
- Gaussian 核。

### 5.6 `kernels/pckKernelExponential.m`
功能：
- Exponential 核。
- APCK 可用，GE 路径不推荐。

### 5.7 `kernels/pckKernelMatern32.m`
功能：
- Matern 3/2 核。

相关设置字段：

```matlab
ModelOpts.KrigingOpts.corr = 'gaussian';  % Gaussian 核
ModelOpts.KrigingOpts.corr = 'exponential'; % Exponential 核
ModelOpts.KrigingOpts.corr = 'matern32';  % Matern 3/2 核
```

## 6. `predict/`

### 6.1 `aPC_Predict.m`
功能：
- 统一预测入口。
- 根据 `results.ModelType` 自动分发到 APC / APCK / GE / GE low-rank 预测。

调用位置：
- `main.m`
- `main_compare.m`

### 6.2 `aPC_PCKPredict.m`
功能：
- APCK / GEAPCK 预测。

### 6.3 `aPC_PCKPredictLowRank.m`
功能：
- GE low-rank 预测。
- 使用训练阶段保存的低秩梯度模态直接构造交叉协方差。

相关设置字段：
- 预测阶段主要读取训练结果，不直接重新读 `ModelOpts` 做大分支。

## 7. `utils/`

### 7.1 `aPC_PrepareUGradients.m`
功能：
- 把物理空间梯度 `dY/dX` 转成标准化空间梯度 `dY/dU`。

相关设置字段：

```matlab
ModelOpts.GradientMode = 'provided';      % 直接使用 data.dYdX_train
ModelOpts.GradientMode = 'finite-difference'; % 调用 ModelFcn 做差分梯度
ModelOpts.GradientMode = 'auto';          % 优先 provided，失败再回退
```

### 7.2 `aPC_PrepareModelGradients.m`
功能：
- 把 `dY/dU` 进一步映射到降维后的 model space。
- GEAPCK / GE low-rank 训练前都会经过这里。

### 7.3 `aPC_FDGradients.m`
功能：
- 数值梯度计算。

相关设置字段：

```matlab
ModelOpts.FDOpts.type = 'forward';        % 前向差分
ModelOpts.FDOpts.type = 'central';        % 中心差分
ModelOpts.FDOpts.step = 1e-6;             % 相对扰动步长
data.ModelFcn = @my_model;                % 原模型函数句柄
```

### 7.4 `aPC_PCE_loo_error.m`
功能：
- APC 路径的 LOO 误差辅助函数。

### 7.5 `aPC_check_orthogonality.m`
功能：
- 检查基函数数值正交性。

### 7.6 `aPC_compute_basis_norms.m`
功能：
- 计算基函数范数。

### 7.7 `aPC_print.m`
功能：
- 打印训练结果摘要。

调用位置：
- `main.m`
- `main_compare.m`

### 7.8 `aPC_printUQ.m`
功能：
- 打印 UQ 结果。

### 7.9 `aPC_printSensitivity.m`
功能：
- 打印灵敏度分析结果。

## 8. `dimreduct/`

### 8.1 `aPC_dimReduction.m`
功能：
- 降维总入口。
- 根据 `ReductionMethod` 分发到具体降维器。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'pls';        % 降维方法
ModelOpts.red_dim = 4;                    % 目标维数
```

### 8.2 `aPC_dimReduction_pls.m`
功能：
- PLS 降维。
- 当前最常用、最稳妥的降维选择之一。

### 8.3 `aPC_dimReduction_gepls.m`
功能：
- GEPLS 降维。
- 降维阶段本身就会使用梯度。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'gepls';      % 使用 GEPLS
ModelOpts.gepls_opts.gamma = 0.3;         % 方法专属参数示例
```

### 8.4 `aPC_dimReduction_pca.m`
功能：
- PCA 降维。

### 8.5 `aPC_dimReduction_kpca.m`
功能：
- KPCA 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'kpca';       % 使用 KPCA
ModelOpts.kpca_opts.kernel_type = 'gaussian'; % 输入核类型
ModelOpts.kpca_opts.kernel_gamma = 0.1;   % 核尺度参数
ModelOpts.kpca_opts.kernel_degree = 2;    % 多项式核次数
ModelOpts.kpca_opts.kernel_offset = 1;    % 多项式核偏置项
```

### 8.6 `aPC_dimReduction_spca_hsic.m`
功能：
- 监督 PCA-HSIC 降维。

### 8.7 `aPC_dimReduction_skpca_hsic.m`
功能：
- 监督 KPCA-HSIC 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'skpca_hsic'; % 使用监督 KPCA-HSIC
ModelOpts.skpca_opts.kernel_type = 'gaussian'; % 输入核类型
ModelOpts.skpca_opts.label_kernel_type = 'gaussian'; % 标签核类型
```

### 8.8 `aPC_dimReduction_mcp.m`
功能：
- MCP 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'mcp';        % 使用 MCP
ModelOpts.mcp_opts.maxIter = 3000;        % 最大迭代次数
ModelOpts.mcp_opts.whiten = 1;            % 是否先白化
ModelOpts.mcp_opts.alpha_cov = 1.0;       % 协方差项权重
ModelOpts.mcp_opts.lambda_y = 0.05;       % 监督项权重
```

### 8.9 `aPC_dimReduction_ica.m`
功能：
- ICA 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'ica';        % 使用 ICA
ModelOpts.ica_opts.approach = 'symm';     % 求解方式
ModelOpts.ica_opts.fun = 'tanh';          % 非线性函数
ModelOpts.ica_opts.maxIter = 200;         % 最大迭代次数
```

### 8.10 `aPC_dimReduction_rp.m`
功能：
- 随机投影降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'rp';         % 使用随机投影
ModelOpts.rp_opts.type = 'gaussian';      % 随机矩阵类型
ModelOpts.rp_opts.scale = 'jl';           % 是否按 JL 缩放
ModelOpts.rp_opts.seed = 80;              % 随机种子
```

### 8.11 `aPC_dimReduction_as.m`
功能：
- Active Subspace 降维。

### 8.12 `aPC_dimReduction_as_pls.m`
功能：
- AS + PLS 组合降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'as_pls';     % 使用 AS-PLS
ModelOpts.as_pls_opts.as_dim = 25;        % AS 阶段的中间维数
```

### 8.13 `aPC_dimReduction_sas.m`
功能：
- SAS 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'sas';        % 使用 SAS
ModelOpts.sas_opts.weight_method = 'zpower'; % 权重方式示例
```

### 8.14 `aPC_dimReduction_sgs.m`
功能：
- SGS 降维。

相关设置字段：

```matlab
ModelOpts.ReductionMethod = 'sgs';        % 使用 SGS
ModelOpts.sgs_opts.alpha = 0.2;           % 监督项权重
ModelOpts.sgs_opts.bandwidth = [];        % 核带宽
ModelOpts.sgs_opts.eps_reg = 1e-12;       % 正则项
ModelOpts.sgs_opts.normalize = true;      % 是否归一化
ModelOpts.sgs_opts.centerU = false;       % 是否中心化 U
```

### 8.15 `KPCA/Kernel.m`
功能：
- KPCA 用的核函数工具类。

### 8.16 `KPCA/KernelPCA.m`
功能：
- KPCA 主实现。

### 8.17 `KPCA/KernelPCAOption.m`
功能：
- KPCA 配置对象。

### 8.18 `KPCA/KernelPCAVisualization.m`
功能：
- KPCA 可视化辅助。

### 8.19 `SKPCA/SupervisedKernelPCA.m`
功能：
- 监督 KPCA 主实现。

### 8.20 `SKPCA/SupervisedKernelPCAOption.m`
功能：
- 监督 KPCA 配置对象。

## 9. `benchmark/`

这一层主要分成三类文件：benchmark 函数本体、输入分布文件、解析梯度文件。

### 9.1 benchmark 函数本体

#### 9.1.1 `ishigami.m`
功能：
- Ishigami 基准函数。

#### 9.1.2 `wingweight.m`
功能：
- Wing Weight 基准函数。

#### 9.1.3 `morris10.m`
功能：
- Morris 10 维基准函数。

#### 9.1.4 `smooth.m`
功能：
- 平滑测试函数。

#### 9.1.5 `rosenbrock.m`
功能：
- Rosenbrock 函数。

#### 9.1.6 `stybtang.m`
功能：
- Styblinski-Tang 函数。

#### 9.1.7 `xiu_nonlinear.m`
功能：
- Xiu 非线性测试函数。

#### 9.1.8 `sobolG.m`
功能：
- Sobol G 函数。

#### 9.1.9 `rackwitz.m`
功能：
- Rackwitz 函数。

#### 9.1.10 `zhou98.m`
功能：
- Zhou 1998 测试函数。

#### 9.1.11 `cornerPeak.m`
功能：
- Corner Peak 函数。

#### 9.1.12 `genz_continuous.m`
#### 9.1.13 `genz_gaussian.m`
功能：
- Genz 类测试函数。

#### 9.1.14 `griewank.m`
功能：
- Griewank 函数。

#### 9.1.15 `chain.m`
#### 9.1.16 `dimN.m`
#### 9.1.17 `dim100.m`
#### 9.1.18 `dim150.m`
#### 9.1.19 `dim200.m`
#### 9.1.20 `dim300.m`
#### 9.1.21 `dim300_pce.m`
#### 9.1.22 `ishigamiHD.m`
功能：
- 各类高维 benchmark。

### 9.2 `inputopts/`
功能：
- 每个 benchmark 对应的 UQLab 输入分布定义。

常见文件：
- `ishigami_Input.m`
- `wingweight_Input.m`
- `morris10_Input.m`
- `dim100_Input.m`
- `dim150_Input.m`
- `dim300_Input.m`

主程序怎么触发：

```matlab
InputOpts = wingweight_Input;              % main.m 中直接指定输入分布
Benchmark.Name = 'wingweight';            % main_compare.m 中通过名字选择
```

### 9.3 `gradients/`
功能：
- benchmark 的解析梯度。

常见文件：
- `gradients/ishigami_grad.m`
- `gradients/wingweight_grad.m`
- `gradients/morris10_grad.m`

主程序怎么触发：

```matlab
data.dYdX_train = wingweight_grad(data.X_train); % 直接提供解析梯度
Benchmark.GradientSource = 'analytic';           % main_compare 中指定解析梯度
```

## 10. `display/`

### 10.1 `aPC_plot.m`
功能：
- 画预测相关图。

### 10.2 `aPC_PlotUQ.m`
功能：
- 画 UQ 结果图。

### 10.3 `aPC_PlotSensitivity.m`
功能：
- 画灵敏度分析图。

调用位置：
- `main.m` 末尾默认注释，可按需打开。

## 11. `UQ/`

### 11.1 `aPC_ComputeUQ.m`
功能：
- 计算 APC 模型的不确定性统计量。

调用位置：
- `main.m` 末尾默认注释，可按需打开。

说明：
- 当前主要面向 APC 使用。

## 12. `sensitivity/`

### 12.1 `aPC_ComputeSensitivity.m`
功能：
- 计算 APC 模型的 Sobol 灵敏度指数。

调用位置：
- `main.m` 末尾默认注释，可按需打开。

说明：
- 当前主要面向 APC 使用。

## 13. 如何按代码顺序阅读

建议阅读顺序：

1. `core/aPC.m`
2. `main.m`
3. `main_compare.m`
4. `apc_train/aPC_Train.m`
5. `apc_train/aPC_TrainCandidate.m`
6. `basis/aPC_PCE_map_to_U.m`
7. `basis/aPC_GenerateTerms.m`
8. `basis/aPC_PCE_build_Psi.m`
9. `coefficients/aPC_RegressionMethods.m`
10. `coefficients/aPC_PCKBackend.m`
11. `coefficients/aPC_PCKBackendLowRankGE.m`
12. `predict/aPC_Predict.m`
13. `predict/aPC_PCKPredict.m`
14. `predict/aPC_PCKPredictLowRank.m`
15. `utils/aPC_PrepareModelGradients.m`
16. `dimreduct/aPC_dimReduction.m`

按这个顺序看，最容易把“主程序字段怎么控制底层文件”连起来理解。
