function InputOpts = zhou98_Input()
% -------------------------------------------------------------------------
% Input definition for ZHOU (1998) function
% dim100-style: independent Uniform marginals
% -------------------------------------------------------------------------

d = 200;   % 默认维度（你后面可直接复制改成 200 / 300）

for ii = 1:d
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [0 1];
end

end
