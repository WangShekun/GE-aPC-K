function InputOpts = rosenbrock50_Input()
for ii = 1:50
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-1, 1];
end
end
