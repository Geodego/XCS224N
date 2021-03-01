

x = (1:0.3333333:7.6667) + 0.166666;
y = 1 + 0.5*x + randn(1,length(x))*0.08;
hold off; plot(x,y, 'x'); hold on; 
xlabel('x');
ylabel('y');

print -depsc linearData0.eps

for i=1:10,
  val(i) = mean(y(find(i <= x & x <= i+1)));
end;

xx = 1:0.005:8;
yy = val(floor(xx));
plot(xx,yy);

print -depsc staircase0.eps
