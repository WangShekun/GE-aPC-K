function InputOpts = dim100_case1_Input()
% 100D input setting adapted from dim150_Input.

for ii = 1:100
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
