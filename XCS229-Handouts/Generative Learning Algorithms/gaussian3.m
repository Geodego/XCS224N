Sigma = [0.6 0; 0 0.6]; 
meshGaussian(Sigma, mu);
axis tight;
a = axis;

Sigma = [1 0; 0 1];
mu = [1; 0];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_mu1.eps

Sigma = [1 0; 0 1];
mu = [-0.5; 0];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_mu2.eps

Sigma = [1 0; 0 1];
mu = [-1; -1.5];
meshGaussian(Sigma, mu);
axis(a);
print -depsc gaussian_mu3.eps



