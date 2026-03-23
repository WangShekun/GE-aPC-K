function InputOpts = dim150_Input()

for ii = 1:150
    if ii <= 10
        InputOpts.Marginals(ii).Type       = 'Uniform';
        InputOpts.Marginals(ii).Parameters = [0.5 1.5];
    elseif ii <= 20
        InputOpts.Marginals(ii).Type       = 'Lognormal';
        InputOpts.Marginals(ii).Parameters = [1 0.1];
    else
        % 原本 149–150 是 MvNormal，这里退化为独立 Gaussian
        InputOpts.Marginals(ii).Type       = 'Gaussian';
        InputOpts.Marginals(ii).Parameters = [1 0.1];
    end
end
end
