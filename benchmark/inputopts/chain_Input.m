function InputOpts = chain_Input()
d = 300;   % <<< 改这里：60 / 80 / 150 / 200 / 300

for ii = 1:d
    if ii <= 10
        InputOpts.Marginals(ii).Type = 'Uniform';
        InputOpts.Marginals(ii).Parameters = [0.5 1.5];
    elseif ii <= 20
        InputOpts.Marginals(ii).Type = 'Lognormal';
        InputOpts.Marginals(ii).Parameters = [1 0.1];
    else
        InputOpts.Marginals(ii).Type = 'Gaussian';
        InputOpts.Marginals(ii).Parameters = [1 0.1];
    end
end
end
