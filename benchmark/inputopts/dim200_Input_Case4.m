function InputOpts = dim200_Input_Case4()

for ii = 1:200
    if ii <= 10
        InputOpts.Marginals(ii).Type       = 'Uniform';
        InputOpts.Marginals(ii).Parameters = [-1.5 3.5];
    elseif ii <= 20
        InputOpts.Marginals(ii).Type       = 'Lognormal';
        InputOpts.Marginals(ii).Parameters = [1 0.5];
    else
        % 原本 199–200 是 MvNormal，这里退化
        InputOpts.Marginals(ii).Type       = 'Gaussian';
        InputOpts.Marginals(ii).Parameters = [1 0.5];
    end
end
end
