

speed_replay = load('speed_replay.mat').Speed;
speed_theta = load('speed_theta.mat').Speed;
v_ext = load('v_record.mat').v_record;
Speed_replay = mean(speed_replay,2);
Speed_theta = mean(speed_theta,2);
for i = 1:10
    sort_replay = sort(speed_replay(i,:));
    top_n = floor(length(speed_replay(i,:))*0.03);
    thres = sort_replay(end-top_n);
    speed_travel = speed_replay(speed_replay>thres);
    Speed_travel(i) = mean(speed_travel);
end

x = [1,5,9,13];
barwidth = 0.8;
y = [mean(v_ext)*1e3/3, mean(Speed_replay)*1e3/3,mean(Speed_theta)*1e3/3,mean(Speed_travel)*1e3/3];
std_y = [0,std(Speed_replay*1e3/3),std(Speed_theta*1e3/3),std(Speed_travel*1e3/3)];


%% plot figure

% A4 paper size in inches
a4WidthInches = 8.27;
a4HeightInches = 11.69;

% Calculate the desired figure width
figureWidthInches = 0.4 * a4WidthInches;

% Create the figure with the specified width
figure('Units', 'inches', 'Position', [0 0 figureWidthInches 0.8*3/2*figureWidthInches]);

bar(x, y,barwidth, "FaceColor", '#009FB9');
hold on
h = errorbar(x, y, std_y, "Marker","o", "MarkerEdgeColor", '#F18D00', "MarkerFaceColor","#F18D00");
set(h, 'Color', '#F18D00', 'LineWidth', 2, 'LineStyle', '--');

xlim([-1 15]);
ylim([0 9]);
yticks([0,3,6,9]);
xticklabels({'Animal moving','mean replay', 'Theta sweep', 'Top 5% replay'});
xtickangle(10);
ylabel('Speed (m/s)', 'FontName', 'Arial', 'FontSize', 10);

% set gca
set(gca, ...
    'LineWidth', 1, ...
    'XColor', [0,0,0],...
    'YColor', [0,0,0]);

box off;

%save figure
filename = './Figures/Fig7a.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector', 'Resolution', 300)
