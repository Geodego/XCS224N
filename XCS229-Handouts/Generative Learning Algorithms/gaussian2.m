Sigma = [0.6 0; 0 0.6]; 
meshGaussian(Sigma, mu);
a = axis;
print -depsc gaussian_0.6_0_0.6con.eps

Sigma = [1 0; 0 1];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_1_0_1con.eps

Sigma = [2 0; 0 2];
meshGaussian(Sigma, mu); axis(a);
print -depsc gaussian_2_0_2con.eps

Sigma = [1 -0.8; -0.8 1];
meshGaussian(Sigma, mu); axis(a);
print -depsc gaussian_1_-0.8_1con.eps

Sigma = [1 0.5; 0.5 1];
meshGaussian(Sigma, mu); axis(a);
print -depsc gaussian_1_0.5_1con.eps

Sigma = [1 -0.5; -0.5 1];
meshGaussian(Sigma, mu); axis(a);
print -depsc gaussian_1_-0.5_1con.eps

Sigma = [1 0.8; 0.8 1];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_1_0.8_1con.eps

% AZ 24, El 50
Sigma = [1 0.5; 0.5 1];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_1_0.5_1con.eps

Sigma = [3 0.8; 0.8 1];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_3_0.8_1con.eps



Sigma = [0.5 0; 0 0.5]; 
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_0.5_0_0.5con.eps

mu = [0; 0];

contour(log(Z))




