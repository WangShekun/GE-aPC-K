function InputOpts = rosenbrock30_Input()
for ii = 1:30
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-1, 1];
end
end
