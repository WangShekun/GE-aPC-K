function InputOpts = rosenbrock_Input()
d = 200;   % <<< 建议 60–80

for ii = 1:d
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-2 2];
end
end
