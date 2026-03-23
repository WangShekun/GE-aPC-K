function InputOpts = smooth_Input()
d = 200;   % <<< 改这里

for ii = 1:d
    if ii <= 10
        InputOpts.Marginals(ii).Type = 'Uniform';
        InputOpts.Marginals(ii).Parameters = [-1 1];
    elseif ii <= 20
        InputOpts.Marginals(ii).Type = 'Lognormal';
        InputOpts.Marginals(ii).Parameters = [0 0.25];
    else
        InputOpts.Marginals(ii).Type = 'Gaussian';
        InputOpts.Marginals(ii).Parameters = [0 1];
    end
end
end
