x = linspace(0, 1, 1000);
cross_entropy = - log2(x) .* x - log2(1 - x) .* (1 - x);

misclassification = 1 - max(x, (1 - x));

clf
plot(x, cross_entropy)
hold on
%plot(x, misclassification)

xticks([0 0.5 1])
yticks([0 1])

R_p = [0.8 cross_entropy(800)];
R_2 = [1 cross_entropy(999)];
R_1 = [0.6 cross_entropy(600)];

scatter(R_p(1), R_p(2), 50, 'black*')
text(R_p(1), R_p(2) + 0.025, '$\mathbf{L(R_p)}$', 'Interpreter', 'latex', 'FontSize', 15)

scatter(R_1(1), R_1(2), 50, 'black*')
text(R_1(1) - 0.1, R_1(2) - 0.02, '$\mathbf{L(R_1)}$', 'Interpreter', 'latex', 'FontSize', 15)

scatter(R_2(1), R_2(2), 50, 'black*')
text(R_2(1) - 0.13, R_2(2) + 0.025, '$\mathbf{L(R_2)}$', 'Interpreter', 'latex', 'FontSize', 15)

plot([R_1(1) R_2(1)], [R_1(2) R_2(2)])
xlabel('$\hat{p}$', 'Interpreter', 'latex', 'FontSize', 15)
ylabel('cross-entropy loss', 'Interpreter', 'latex', 'FontSize', 15)

R_weighted = [0.8 (R_1(2) + R_2(2)) / 2];
scatter(R_weighted(1), R_weighted(2), 50, 'black*')
text(R_weighted(1) - 0.3, R_weighted(2), '$\mathbf{\frac{|R_1| L(R_1) + |R_2| L(R_2)}{|R_1| + |R_2|}}$', 'Interpreter', 'latex', 'FontSize', 15)

plot([R_weighted(1) R_p(1)], [R_weighted(2) R_p(2)], 'r--', 'LineWidth',2)

export_fig -native crossentropy.eps 
