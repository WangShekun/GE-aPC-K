function y = genz_gaussian(xx, c, w)
% -------------------------------------------------------------------------
% Genz "Gaussian peak" test function (smooth, high-dimensional benchmark)
%
% y(z) = exp( - sum_{i=1}^d c_i^2 (z_i - w_i)^2 ),  z_i in [0,1]
%
% Inputs:
%   xx : N x d
%   c  : 1 x d positive scales (larger -> sharper -> harder)
%   w  : 1 x d shifts in [0,1]
% -------------------------------------------------------------------------

z = xx;
[~, d] = size(z);

if nargin < 2 || isempty(c)
    c = genz_default_c(d);
end
if nargin < 3 || isempty(w)
    w = 0.5*ones(1,d);
end

c = reshape(c,1,[]);
w = reshape(w,1,[]);
if numel(c) ~= d || numel(w) ~= d
    error('c and w must have length d.');
end

y = exp( -sum( ( (z - w) .* c ).^2, 2 ) );
end


function c = genz_default_c(d)
% Mild difficulty for surrogate/PCE at d=60~80
% Too large c makes a very sharp peak -> needs more samples.
idx = 1:d;
c = 1.0 + 0.5*(idx-1)/(d-1);  % c in [1, 1.5]
end
