function InputOpts = ishigami_standard_Input()
for ii = 1:3
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-pi, pi];
end
end
