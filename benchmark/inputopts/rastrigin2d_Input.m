function InputOpts = rastrigin2d_Input()
InputOpts.Marginals(1).Type = 'Uniform';
InputOpts.Marginals(1).Parameters = [-5.12 5.12];
InputOpts.Marginals(2).Type = 'Uniform';
InputOpts.Marginals(2).Parameters = [-5.12 5.12];
end
