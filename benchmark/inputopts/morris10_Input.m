function InputOpts = morris10_Input()
for k = 1:20
    InputOpts.Marginals(k).Type = 'Uniform';
    InputOpts.Marginals(k).Parameters = [0 1];
end
end