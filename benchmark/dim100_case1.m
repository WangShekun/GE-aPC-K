function y = dim100_case1(xx)
% High-dimensional test function adapted from dim150 (d = 100, Case 1)
% xx: N x 100

x = xx;

term1 = sum(x, 2);
term2 = 30 .* x(:,1) .* x(:,2);
term3 = sum(x(:,2:99) .* x(:,3:100), 2);
term4 = sum(sin(x) .* exp(x - 2), 2);

y = term1 + term2 + term3 - term4 - 10;
end
