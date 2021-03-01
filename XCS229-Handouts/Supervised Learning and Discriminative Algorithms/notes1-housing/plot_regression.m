% generates housingWithRegression.eps from matlab's lms solution
%
% dramage 20070921

plot_basic;

Y = prices;
X = [ones(size(sizes)) sizes];

hold on;
theta = (X \ Y)'
xs = linspace(500,5000);
plot(xs, theta(1) + theta(2)*xs, '-');

saveas(gcf, 'housingWithRegression', 'epsc');

