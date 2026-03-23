function y = smooth(xx)
x = xx;
[~, d] = size(x);

alpha = 1.0;
beta  = 0.1;

term1 = sum(log(1 + x.^2), 2);
term2 = beta * sum(x(:,1:d-1).*x(:,2:d), 2);
term3 = alpha * sum(atan(x), 2);

y = term1 + term2 + term3;
end
