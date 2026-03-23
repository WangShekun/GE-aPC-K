function InputOpts = genz_gaussian_Input()
d = 80;

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [0 1];
end
end
