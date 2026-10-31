function InputOpts = oakoh04_Input()
for ii = 1:15
    InputOpts.Marginals(ii).Type = 'Gaussian';
    InputOpts.Marginals(ii).Parameters = [0, 1];
end
end
