function y = rosenbrock(xx)
x = xx;
[~, d] = size(x);

y = sum(100*(x(:,2:d) - x(:,1:d-1).^2).^2 + (1 - x(:,1:d-1)).^2, 2);
end
