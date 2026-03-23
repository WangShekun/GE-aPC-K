function aPC_PlotSensitivity(SA, ModelOpts)
% aPC_PlotSensitivity
% ------------------------------------------------------------
% 绘制 Sobol 灵敏度分析图：
%   (1) 一阶 Sobol 条形图
%   (2) 总 Sobol 条形图
%   (3) 二阶 Sobol 矩阵热力图（diag=一阶, off diag=二阶）
%
% 输入：
%   SA.S_first     - 1 × M 一阶 Sobol
%   SA.S_total     - 1 × M 总 Sobol
%   SA.S_second    - M × M 二阶 Sobol
%   ModelOpts.num_vars - 变量数
%
% ------------------------------------------------------------

M = ModelOpts.num_vars;
S1 = SA.S_first(:)';
St = SA.S_total(:)';
S2 = SA.S_second;

% 构造用于热力图的矩阵 Smat：
%    diagonal = S1(i)
%    off-diagonal = S2(i,j)
Smat = S2;
for i = 1:M
    Smat(i,i) = S1(i);
end

%% ------------ 图形布局 ------------
figure('Name','Sobol Sensitivity Analysis','Color','w','Position',[100 100 1200 800]);

%% ============================================================
% 1) 一阶 Sobol 条形图
%% ============================================================
subplot(2,2,1)
bar(S1,'FaceColor',[0.3 0.5 0.8])
ylabel('First-order Sobol S_i')
xlabel('Input Variable')
title('First-order Sobol Indices')
xticks(1:M); xticklabels(varNames(M));
ylim([0 max(S1)*1.2])
grid on

%% ============================================================
% 2) 总 Sobol 条形图
%% ============================================================
subplot(2,2,2)
bar(St,'FaceColor',[0.8 0.35 0.35])
ylabel('Total Sobol S_{T_i}')
xlabel('Input Variable')
title('Total Sobol Indices')
xticks(1:M); xticklabels(varNames(M));
ylim([0 max(St)*1.2])
grid on

%% ============================================================
% 3) 二阶 Sobol 热力图（改进版）
%% ============================================================
subplot(2,2,[3 4])   % 大一点
imagesc(flipud(Smat));   % flipud 使左下角是 X1-X1
colormap(parula);
colorbar;
title('Second-order Sobol Heatmap (diag = first-order)')
xlabel('Variable j');
ylabel('Variable i');

% 坐标轴标签
xticks(1:M);
yticks(1:M);
xticklabels(varNames(M));
yticklabels(fliplr(varNames(M)));  % Y 轴翻转以对应 flipud

% 在图中写上数值
for ii = 1:M
    for jj = 1:M
        val = Smat(M-ii+1, jj);   % 注意索引flip
        text(jj, ii, sprintf('%.3f', val), ...
            'HorizontalAlignment','center', ...
            'Color','k', 'FontSize', 10)
    end
end

end

%% ==================== 辅助：变量名 =====================
function names = varNames(M)
names = arrayfun(@(i) sprintf('X_%d', i), 1:M, 'UniformOutput', false);
end
