ccc
Speed_replay = load('speed_replay.mat').speed;
Speed_theta = load('speed_theta.mat').speed;
sort_replay = sort(Speed_replay);
top_n = floor(length(Speed_replay)*0.05);
thres = sort_replay(end-top_n);
speed_travel = Speed_replay(Speed_replay>thres);

% histogram(Speed_theta),hold on
% histogram(Speed_replay(Speed_replay<thres))
% histogram(speed_travel(speed_travel<0.5))

x = [1,5,9,13];
barwidth = 0.8;
y = [1.4/3, mean(Speed_replay)*1e3/3,mean(Speed_theta)*1e3/3,mean(speed_travel)*1e3/3];
std_y = [0,std(Speed_replay*1e3/3),...
    std(Speed_theta*1e3/3),...
    std(speed_travel*1e3/3)];

bar(x,y,barwidth),hold on
h = errorbar(x, y, std_y, 'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue');
set(h, 'Color', 'red');
set(h, 'LineWidth', 2);
xlim([-1 15])

xticklabels({'Animal moving speed','All replays', 'Theta sweeps', 'Top 5% of replays'});

% 设置坐标轴的字体和大小
set(gca, 'FontName', 'Arial', 'FontSize', 15);
ylabel('Average speed(m/s)', 'FontName', 'Arial', 'FontSize', 14);
% 设置 x 轴的线条粗细
set(gca, 'LineWidth', 1.5);
