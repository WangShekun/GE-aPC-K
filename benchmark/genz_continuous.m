function y = genz_continuous(xx, c, w)
% -------------------------------------------------------------------------
% Genz "Continuous" test function (smooth-ish, high-dimensional benchmark)
%
% y(z) = exp( - sum_{i=1}^d c_i * |z_i - w_i| ), z_i in [0,1]
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

y = exp( -sum( abs(z - w) .* c, 2 ) );
end


function c = genz_default_c(d)
% Mild difficulty for d=60~80
idx = 1:d;
c = 0.5 + 1.0*(idx-1)/(d-1);  % c in [0.5, 1.5]
end
