% Generates housingData.eps
%
% dramage 2007

plot_setup

clf;
plot(sizes, prices, 'x');
title('housing prices');
xlabel('square feet');
ylabel('price (in $1000)');
axis([500 5000 -50 1050]);

saveas(gcf, 'housingData', 'epsc');
