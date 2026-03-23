function y = dim300(xx)
% High-dimensional test function (d = 300, Case 5)

x = xx;

term1 = sum(x, 2);
term2 = 30 .* (x(:,1).^2) .* (x(:,2).^2);
term3 = sum(x(:,2:300).^2 .* x(:,1:299).^2, 2);
term4 = sum(sin(x) .* exp(x - 2), 2);

y = term1 + term2 + term3 - term4 + 10;
end
