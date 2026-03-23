function InputOpts = cornerPeak_Input(d)
% -------------------------------------------------------------------------
% Input definition for Corner-Peak function
% -------------------------------------------------------------------------

if nargin < 1
    d = 200;   % 论文中的一个默认低维示例
end

for ii = 1:d
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [0 1];
end

end
