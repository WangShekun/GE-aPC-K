function y = chain(xx)
% Chain-type high-dimensional function
x = xx;
[~, d] = size(x);

alpha = 2.0;
beta  = 0.5;

term1 = sum(x, 2);
term2 = sum(x(:,1:d-1) .* x(:,2:d), 2);
term3 = sum(tanh(x), 2);

y = term1 + alpha*term2 + beta*term3;
end
