function [Y] = dim100(X)

X = transpose(X);

% add some increasing weight to the various components of X
L = repmat(transpose(1:size(X,1)), 1, size(X,2));

% create an array that is a non-linear sum of the elements of the original
% X input vector
Y = 300 + L.*X.^3 + 1/3*L.*log(X.^2 + X.^4) -5*L.*X ;

% squeeze it to a single row, and add extra sensitivity terms for
% X51
% X1X2^2
% X50X54^2
% X3X5
% X2X4

Y = mean(Y,1) + X(1,:).*X(2,:).^2 - X(5,:).*X(3,:) + X(2,:).*X(4,:) + X(51,:) + X(50,:).*X(54,:).^2;
Y = transpose(Y);
end
