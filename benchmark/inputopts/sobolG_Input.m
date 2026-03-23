function InputOpts = sobolG_Input()
% -------------------------------------------------------------------------
% Input definition for Sobol' g-function
% -------------------------------------------------------------------------

n = 200;

for ii = 1:n
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [0 1];
end

end
