% generates housingGD*.eps using gradient descent
%
% dramage 20070921

% load data
plot_setup

% step size
alpha = 4 * [10^-4 10^-10];

% initial parameter guess
theta = [0 0];

% dataset
Y = prices;
X = [ones(size(sizes)) sizes];

for stop=[0 2 4 6 8 10 5000];

  for step=1:stop;
    % batch gradient
    g = (Y-X*theta')'*X;
    theta = theta + alpha .* g;
  end
  
  theta

  plot_basic;
  hold on;
  xs = linspace(500,5000);
  plot(xs,theta(1) + theta(2)*xs, '-');
  saveas(gcf, sprintf('housingGD-%d',stop), 'epsc');
  
  pause(1);
end

saveas(gcf, 'housingGDConverged', 'epsc');
