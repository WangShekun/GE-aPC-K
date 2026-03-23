function y = dim150(xx)
% High-dimensional test function (d = 150, Case 1)
% xx: N x 150

x = xx;

term1 = sum(x, 2);
term2 = 30 .* x(:,1) .* x(:,2);
term3 = sum(x(:,2:149) .* x(:,3:150), 2);
term4 = sum(sin(x) .* exp(x - 2), 2);

y = term1 + term2 + term3 - term4 - 10;
end

