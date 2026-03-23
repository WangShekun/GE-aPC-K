function aPC_PlotUQ(UQ, Data)
% aPC_PlotUQ
% ------------------------------------------------------------
% 高级可视化：专属于 Uncertainty Quantification，不重复模型诊断图
%
% 包括：
%   1) Quantile Fan Chart（分位数扇形图）
%   2) Uncertainty Band（置信区间带）
%   3) Tail Profile（尾部剖面图）
%   4) Risk Rose Plot（风险风玫瑰图）
%   5) Uncertainty Waterfall（不确定性瀑布图）
%
% 输入：
%   UQ.mean, UQ.std, UQ.CI95, UQ.VaR95, UQ.CVaR95 ...
%   Data.Y_pred
% ------------------------------------------------------------

% 如果没有预测样本，则跳过绘图
if nargin < 2 || ~isfield(Data,'Y_pred') || isempty(Data.Y_pred)
    return;
end

y = Data.Y_pred(:);
N = length(y);

%% ------------------------------------------------------------
% 1) Quantile Fan Chart
% ------------------------------------------------------------
figure('Name','UQ - Quantile Fan Chart','Color','w');
hold on; box on; grid on;

qs = linspace(0.01,0.99,15);
qv = quantile(y, qs);

colors = parula(length(qs));

for i = 1:length(qs)
    plot([0 1],[qv(i) qv(i)],'LineWidth',3,...
        'Color',colors(i,:));
end

title('Quantile Fan Chart (Uncertainty Structure)');
xlabel('Normalized Index');
ylabel('Value');
ylim([min(y) max(y)]);

cb = colorbar;
cb.Ticks = linspace(0,1,5);
cb.TickLabels = {'1%','25%','50%','75%','99%'};
ylabel(cb,'Quantiles');

%% ------------------------------------------------------------
% 2) Uncertainty Band（CI95 区间带 + 分位带）
% ------------------------------------------------------------
figure('Name','UQ - Uncertainty Band','Color','w'); hold on; box on; grid on;

xs = linspace(0,1,N);

% 95% 区间
fill([xs fliplr(xs)], ...
     [repmat(UQ.CI95(1),1,N) fliplr(repmat(UQ.CI95(2),1,N))], ...
     [0.8 0.9 1], 'FaceAlpha',0.4, 'EdgeColor','none');

% 中位数
q50 = quantile(y,0.5);
plot(xs, repmat(q50,1,N), 'k-', 'LineWidth',2);

% ±std
plot(xs, repmat(UQ.mean + UQ.std,1,N), 'r--','LineWidth',1.5);
plot(xs, repmat(UQ.mean - UQ.std,1,N), 'r--','LineWidth',1.5);

title('Uncertainty Band (CI95 + Std + Median)');
xlabel('Normalized Index');
ylabel('Value');

legend('CI95','Median','Mean ± Std','Location','best');

%% ------------------------------------------------------------
% 3) Tail Profile（尾部剖面图，极值敏感度）
% ------------------------------------------------------------
figure('Name','UQ - Tail Profile','Color','w');
hold on; box on; grid on;

[f,x] = ksdensity(y,'NumPoints',400);

plot(x,f,'b','LineWidth',2);

% VaR 和 CVaR 竖线
xline(UQ.VaR95,'r--','LineWidth',2,'Label','VaR95');
xline(UQ.CVaR95,'m--','LineWidth',2,'Label','CVaR95');

title('Tail Risk Profile (VaR & CVaR)');
xlabel('Value'); ylabel('Density');

%% ------------------------------------------------------------
% 4) Risk Rose Plot（尾部风险方位图）
% ------------------------------------------------------------
figure('Name','UQ - Risk Rose Plot','Color','w');
polaraxes; hold on;

theta = linspace(0,2*pi,N);
radius = abs(y - UQ.mean);       % 离均值越远风险越大
radius = radius / max(radius);   % 归一化

polarplot(theta, radius,'LineWidth',1.5);

title('Risk Rose Plot (Deviation from Mean)');

%% ------------------------------------------------------------
% 5) Uncertainty Waterfall（贡献瀑布图）
% ------------------------------------------------------------
figure('Name','UQ - Uncertainty Waterfall','Color','w');
hold on; box on; grid on;

vals = [UQ.std, UQ.CVaR95-UQ.mean];
names = {'Std (Total Spread)','CVaR Tail Risk'};

bar(vals,'FaceColor',[0.3 0.6 0.9]);

set(gca,'XTickLabel',names,'XTickLabelRotation',15);
ylabel('Magnitude');
title('Uncertainty Waterfall');

end
