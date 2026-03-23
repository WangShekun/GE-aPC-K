function y =ishigamiHD(xx)
x = xx;
[~, d] = size(x);

a = 7; 
b = 0.1;
c = 0.05;

y = sin(x(:,1)) ...
  + a*sin(x(:,2)).^2 ...
  + b*(x(:,3).^4).*sin(x(:,1)) ...
  + c*sum(x(:,4:d),2);
end
