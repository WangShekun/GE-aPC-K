function InputOpts = dim200_Input_Case2()

for ii = 1:200
    InputOpts.Marginals(ii).Type       = 'Gaussian';
    InputOpts.Marginals(ii).Parameters = [1 0.5];
end
end
