% Define input data.
data = [
    [0.1 0.8 1]
    [0.3 0.7 1]
    [0.6 0.75 1]
    [0.8 0.9 1]
    [0.9 0.8 1]
    [0.2 0.2 0]
    [0.05 0.25 0]
    [0.35 0.3 0]
    [0.4 0.35 1]
    [0.45 0.6 1]
    [0.5 0.1 1]
    [0.6 0.6 0]
    [0.65 0.45 0]
    [0.8 0.5 0]
    [0.6 0.2 1]
    [0.8 0.2 1]
    [0.95 0.1 1]
    ];


% Split out data and labels, positive and negatives.
X = data(:, 1:2);
y = data(:, 3);

pos = X(y == 1, :);
neg = X(y ~= 1, :);

% Display.
clf
s1 = scatter(pos(:, 1), pos(:, 2), 100, 'g', '+');
hold on
s2 = scatter(neg(:, 1), neg(:, 2), 100, 'r', 'o');

xlabel('$x_1$', 'Interpreter', 'Latex')
ylabel('$x_2$', 'Interpreter', 'Latex')
set(gca, 'FontName', 'Times')
set(gcf, 'Color', 'w');
xlim([0 1])
ylim([0 1])

export_fig -native boosting-step0.eps



x1_thresh = 0.375;
h1 = line([x1_thresh x1_thresh], [1 0], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');

misclass_neg = neg(:, 1) > x1_thresh;
misclass_pos = pos(:, 1) < x1_thresh;


s3 = scatter(neg(misclass_neg, 1), neg(misclass_neg, 2), 300, 'black', 'square', 'LineWidth', 2);
s4 = scatter(pos(misclass_pos, 1), pos(misclass_pos, 2), 300, 'black', 'square', 'LineWidth', 2);

export_fig -native boosting-step1.eps

delete([h1, s1, s2, s3, s4])

s1 = scatter(neg(misclass_neg, 1), neg(misclass_neg, 2), 300, 'r', 'o', 'LineWidth', 2);
s2 = scatter(pos(misclass_pos, 1), pos(misclass_pos, 2), 300, 'g', '+', 'LineWidth', 2);
s3 = scatter(neg(~misclass_neg, 1), neg(~misclass_neg, 2), 50, 'r', 'o');
s4 = scatter(pos(~misclass_pos, 1), pos(~misclass_pos, 2), 50, 'g', '+');

x2_thresh = 0.65;
misclass_neg = neg(:, 2) > x2_thresh;
misclass_pos = pos(:, 2) < x2_thresh;

s5 = scatter(pos(misclass_pos, 1), pos(misclass_pos, 2), 300, 'black', 'square', 'LineWidth', 2);


h1 = line([0 1], [x2_thresh x2_thresh], 'LineWidth', 2, 'LineStyle','--', 'Color', 'black');


export_fig -native boosting-step2.eps