
equator_lim = 15;

% Generate some uniformly sampled data.
% num_ex = 100;
% data = rand(num_ex, 2);
% data(:, 1) = data(:, 1) * 12;
% data(:, 2) = data(:, 2) * 180 - 90;
% data = [data, ones(size(data, 1), 1)];
% save('ski.mat', 'data')

% Use pre-generated data.
load('ski.mat', 'data');

% Define negative regions
equator = data(:, 2) < 15 & data(:, 2) > -equator_lim;
data(equator, 3) = 0;

summer_north = data(:, 1) > 3 & data(:, 1) < 9 & data(:, 2) > equator_lim;
data(summer_north, 3) = 0;

summer_south = (data(:, 1) <= 4 | data(:, 1) >= 8) & data(:, 2) < -equator_lim;
data(summer_south, 3) = 0;

% Split out data and labels, positive and negatives.
X = data(:, 1:2);
y = data(:, 3);

pos = X(y == 1, :);
neg = X(y ~= 1, :);

% Display.
clf
scatter(pos(:, 1), pos(:, 2), 50, 'g', '+')
hold on
scatter(neg(:, 1), neg(:, 2), 50, 'r', 'o')

axis([-0.5 12.5 -92 92])
xticks(linspace(0, 12, 13))
yticks([-90, 0, 90])
xlabel('Time (months)')
ylabel('Latitude (degrees)')
set(gca, 'FontName', 'Times')
set(gcf, 'Color', 'w');
export_fig -native ski.eps


h11 = line([-0.5 3], [15 15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h12 = line([3 3], [15 92], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

h21 = line([9 12.5], [15 15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h22 = line([9 9], [15 92], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

h31 = line([4 8], [-15 -15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h32 = line([4 4], [-15 -92], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h33 = line([8 8], [-15 -92], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

export_fig -native ski-region.eps 

delete([h11, h12, h21, h22, h31, h32, h33])

h1 = line([-0.5 12.5], [15 15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
t1 = text(6.5, -40, '$\mathbf{R_1}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);
t2 = text(6.5, 50, '$\mathbf{R_2}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);

export_fig -native ski-split1.eps 

delete(t2)
h2 = line([3 3], [15 92], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
t21 = text(1.5, 50, '$\mathbf{R_{21}}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);
t22 = text(7.5, 50, '$\mathbf{R_{22}}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);

export_fig -native ski-split2.eps 

h3 = line([-0.5 12.5], [-15 -15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
delete(t1)
t11 = text(6.5, 0, '$\mathbf{R_{12}}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);
t12 = text(6.5, -50, '$\mathbf{R_{11}}$', 'Interpreter', 'latex',  'HorizontalAlignment', 'center', 'FontSize', 30);


export_fig -native ski-split3.eps 