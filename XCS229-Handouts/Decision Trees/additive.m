% Generate some uniformly sampled data.
% num_ex = 100;
% data = rand(num_ex, 2);
% data = [data, ones(size(data, 1), 1)];
% save('additive.mat', 'data')

% Use pre-generated data.
load('additive.mat', 'data');

% Define negative regions
data(data(:, 1) +  data(:, 2) < 1, 3) = 0;

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

xlabel('$x_1$', 'Interpreter', 'Latex')
ylabel('$x_2$', 'Interpreter', 'Latex')
set(gca, 'FontName', 'Times')
set(gcf, 'Color', 'w');

h1 = line([0 1], [1 0], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

export_fig -native additive_linear.eps

delete(h1)

h1 = line([0.7 0.7], [1 0], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h2 = line([0.7 0], [0.4 0.4], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h3 = line([0.3 0.3], [0.4 1], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h4 = line([0.7 1], [0.15 0.15], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');
h5 = line([0 0.3], [0.8 0.8], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

export_fig -native additive_dt.eps
