function aPC_plot(data, results, ModelOpts, varargin)
% aPC_plot - 高质量可视化中心：带漂亮风格的 aPC 图形集
%
% 用法：
%   aPC_plot(data,results,ModelOpts)                    % 默认
%   aPC_plot(data,results,ModelOpts,"pdfcdf","input")
%   aPC_plot(data,results,ModelOpts,"all")
%
% 可选模块：
%   "input"         - 输入样本分布（直方图 + ECDF）
%   "pdfcdf"        - 输出 PDF & CDF
%   "exceedance"    - 超越概率图
%   "diagnostics"   - 预测诊断图
%   "orthogonality" - 正交性 Gram 矩阵图
%   "coefficients"  - PCE 系数
%
% 特殊选项：
%   "all"           - 绘制所有图
%

all_modules = ["input","pdfcdf","exceedance","diagnostics","orthogonality","coefficients","coeff_scatter"];
default_modules = ["pdfcdf","exceedance","diagnostics"];

%% 解析模块选项
if nargin < 4
    modules = default_modules;
else
    args = lower(string(varargin));
    if any(args=="all")
        modules = all_modules;
    else
        modules = intersect(args, all_modules);
    end
end

%% 调用绘制模块
if any(modules=="input"),         plot_input(data.X_train); end
if any(modules=="pdfcdf"),        plot_pdfcdf(data.Y_val,data.Y_pred); end
if any(modules=="exceedance"),    plot_exceedance(data.Y_val,data.Y_pred); end
if any(modules=="diagnostics"),   plot_diagnostics(data.Y_val,data.Y_pred); end
if any(modules=="orthogonality"), plot_orthogonality(results,ModelOpts); end
if any(modules=="coefficients"),  plot_coeffs(results); end
if any(modules=="coeff_scatter"), plot_coeffs_scatter(results); end


end  % ===== 主函数结束 =====



%% ============================================================
% 子函数：漂亮样式的 Input Distribution（使用你原来的风格）
%% ============================================================
function plot_input(samples)

[N,d] = size(samples);
nbins = max(25, round(sqrt(N)));
palette = lines(d+2);

ncols = 2;
nrows = ceil(d/ncols);

figure('Name','Input Sample Distributions','Color','w');

for i = 1:d
    subplot(nrows,ncols,i);
    xi = samples(:,i);

    col_hist = palette(i,:);
    col_ecdf = palette(i+1,:);

    yyaxis left
    histogram(xi, nbins, 'FaceColor', col_hist, ...
              'EdgeColor','k', 'FaceAlpha', 0.65);
    ylabel('Frequency');
    grid on

    yyaxis right
    [F,X] = ecdf(xi);
    plot(X,F,'LineWidth',1.8,'Color',col_ecdf);
    ylabel('ECDF'); ylim([0 1]);

    title(sprintf('Input Dimension %d',i));
    legend({'Histogram','ECDF'}, 'Location','best');
end
end



%% ============================================================
% 子函数：漂亮的 PDF + CDF 对比
%% ============================================================
function plot_pdfcdf(y_true, y_pred)
figure('Name','Output PDF & CDF Comparison','Color','w');

subplot(1,2,1)
[f1,x1] = ksdensity(y_true);
[f2,x2] = ksdensity(y_pred);
plot(x1,f1,'b-','LineWidth',1.8); hold on;
plot(x2,f2,'r--','LineWidth',1.8);
legend('True PDF','aPC PDF');
xlabel('y'); ylabel('Density');
title('PDF Comparison');
grid on;

subplot(1,2,2)
[Ft,Xt] = ecdf(y_true);
[Fp,Xp] = ecdf(y_pred);
plot(Xt,Ft,'b-','LineWidth',1.8); hold on;
plot(Xp,Fp,'r--','LineWidth',1.8);
legend('True CDF','aPC CDF');
xlabel('y'); ylabel('CDF');
title('CDF Comparison');
grid on;
end



%% ============================================================
% 子函数：漂亮的 Exceedance 超越图
%% ============================================================
function plot_exceedance(y_true, y_pred)

figure('Name','Exceedance Plot (Tail Accuracy)','Color','w');

[F_true, X_true] = ecdf(y_true);
[F_pred, X_pred] = ecdf(y_pred);

plot(X_true,1-F_true,'b-','LineWidth',1.8); hold on;
plot(X_pred,1-F_pred,'r--','LineWidth',1.8);

set(gca,'YScale','log');
xlabel('y'); ylabel('P(Y > y)');
title('Exceedance Comparison');
legend('True','aPC','Location','southwest');
grid on;

end



%% ============================================================
% 子函数：预测诊断图（使用更漂亮的风格）
%% ============================================================
function plot_diagnostics(y_true, y_pred)
res = y_pred - y_true;
absres = abs(res);
eps0 = 1e-15;                % 防 log(0)
absres(absres<eps0) = eps0;

logr = log10(absres);        % 主角

N = length(res);

figure('Name','Residual Diagnostics (log scale)','Color','w');

%% (1) log10(|res|) vs index
subplot(1,3,1)
plot(1:N, logr,'k.','MarkerSize',6);
xlabel('Sample index');
ylabel('log_{10}(|Residual|)');
title('Residual magnitude vs sample');
grid on;

%% (2) PDF of log10(|res|)
subplot(1,3,2)
[f,x] = ksdensity(logr);
plot(x,f,'r-','LineWidth',1.6); hold on;
histogram(logr,40,'Normalization','pdf',...
          'FaceColor',[0.3 0.4 0.8],'FaceAlpha',0.55);
xlabel('log_{10}(|Residual|)');
ylabel('PDF');
title('Residual magnitude PDF');
grid on;

%% (3) CDF of log10(|res|)
subplot(1,3,3)
[F,X] = ecdf(logr);
plot(X,F,'b-','LineWidth',1.6);
xlabel('log_{10}(|Residual|)');
ylabel('CDF');
title('Residual magnitude CDF');
grid on;

end

%% ============================================================
% 子函数：正交性检查（使用 Gram 矩阵热力图）
%% ============================================================
function plot_orthogonality(results, ModelOpts)

univ = results.univ_vals;
P    = ModelOpts.Order;
M    = size(univ,2);

% 自动布局：接近正方形
nrow = ceil(sqrt(M));
ncol = ceil(M / nrow);

figure('Name','Orthogonality Check','Color','w');

for j = 1:M
    phij = squeeze(univ(:,j,1:P+1));
    G = (phij' * phij) / size(phij,1);

    subplot(nrow, ncol, j);
    imagesc(G);
    colormap(parula);
    axis equal tight;

    title(sprintf('Gram Matrix - Dim %d', j), 'FontSize', 12);

    % 让 colorbar 占空间更合理
    cb = colorbar;
    cb.FontSize = 10;
end

% 让整体布局更紧凑
set(gcf);  % 控制图窗大小
sgtitle('Orthogonality Gram Matrices','FontSize',14);

end


%% ============================================================
% 子函数：系数图（美化过）
%% ============================================================
function plot_coeffs(results)
coef = results.coefficients(:);

% 去掉零系数（完全不画）
nz = abs(coef) > 0;  
coef = coef(nz);

% 避免 log(0)
eps0 = 1e-16;
abscoef = abs(coef);
abscoef(abscoef<eps0) = eps0;

% 排序
[abs_sorted, idx] = sort(abscoef,'descend');
log_abs_sorted = log10(abs_sorted);

figure('Name','PCE Coefficients (clean)','Color','w');

stem(log_abs_sorted,'filled','MarkerSize',3,'Color',[0.1 0.2 0.6]); 
grid on;

xlabel('Sorted index (non-zero coefficients only)');
ylabel('log_{10}(|Coefficient|)');
title('Non-zero PCE coefficients (sorted, log-scale)');

end

%% ============================================================
% 子函数：稀疏系数散点图（仅绘制非零系数）
%  - 纵轴为对数尺度
%  - 点大小 ~ log(|c|)
%  - 同阶数同色
%% ============================================================
function plot_coeffs_scatter(results)

coef_all = results.coefficients(:);   % 全部系数
Indices  = results.Indices;           % 多索引矩阵 (K × M)

% —— 过滤非零系数 —— 
nz = abs(coef_all) > 0;   % 稀疏约束选择出的项（LARS/Lasso）
coef    = coef_all(nz);
Ind_nz  = Indices(nz, :);
K_nz    = length(coef);

if K_nz == 0
    warning('No non-zero coefficients to plot.');
    return;
end

% —— log(|c|) —— 
eps0    = 1e-16;
abscoef = abs(coef);
abscoef(abscoef < eps0) = eps0;
logc    = log10(abscoef);

% —— 阶数 = L1 范数 —— 
orders = sum(Ind_nz, 2);  % 每个基函数的总阶
pmax   = max(orders);

% —— 配色（同阶同色） —— 
cmap   = lines(pmax+1);
colors = cmap(orders+1, :);

% —— 点大小随 log|c| —— 
smin = 15;
smax = 120;
log_min = min(logc);
log_max = max(logc);

if log_max - log_min < 1e-12
    sz = smin * ones(K_nz,1);
else
    sz = smin + (logc - log_min) ./ (log_max - log_min) * (smax - smin);
end

% —— 绘图 —— 
figure('Name','Sparse PCE Coefficients (Scatter)','Color','w');

scatter(1:K_nz, abscoef, sz, colors, ...
        'filled', 'MarkerFaceAlpha', 0.85);

set(gca,'YScale','log');
xlabel('Index of Non-zero Terms');
ylabel('|Coefficient| (log scale)');
title('Sparse PCE Coefficients (non-zero only)');
grid on;

% —— 图例（每个阶数一个颜色） —— 
hold on;
uniq_ord = unique(orders);

for oi = uniq_ord'
    scatter(nan, nan, 60, cmap(oi+1,:), 'filled');
end

legend(arrayfun(@(o) sprintf('Order %d', o), uniq_ord, 'UniformOutput',false), ...
       'Location','northeastoutside');

end
