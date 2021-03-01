% outputs basic stats about the data using matlabs lms
%
% dramage 20070921


plot_setup

% basic statistics
Y = prices;
X = [ones(size(sizes)) sizes];
theta = (X\Y)'   % [95.7772    0.1286]

Y = prices;
X = [ones(size(sizes)) sizes rooms];
theta = (X\Y)'   % [154.1185    0.1408  -26.3240]
