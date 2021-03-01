
function meshGaussian(Sigma, mu);

hold off;
Z = zeros(50,50);
x1=zeros(50,1);x2=x1;
for i=1:50, for j=1:50,
  x1(i) = 1.3*(i-25)/10; x2(j) = 1.3*(j-25)/10; 
  x = [x1(i); x2(j)]; 
  Z(i,j) = 1/2/pi/sqrt(det(Sigma)) * exp(-(x-mu)'*inv(Sigma)*(x-mu));
end; end;
%mesh(x1, x2, Z);
contour(x1, x2, Z);


