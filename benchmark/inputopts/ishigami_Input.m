function InputOpts = ishigami_Input()
InputOpts.Marginals(1).Type       = 'Gaussian';
InputOpts.Marginals(1).Parameters = [3 0.2];
InputOpts.Marginals(2).Type       = 'Uniform';
InputOpts.Marginals(2).Parameters = [0 2];
InputOpts.Marginals(3).Type       = 'Gaussian';
InputOpts.Marginals(3).Parameters = [3 0.1];
end