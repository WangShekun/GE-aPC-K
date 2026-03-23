function InputOpts = dim100_Input()
for ii = 1:100
    InputOpts.Marginals(ii).Type = 'Uniform';
    if ii == 20
        InputOpts.Marginals(ii).Parameters = [1 3];
    else
        InputOpts.Marginals(ii).Parameters = [1 2];
    end
end
end

% function InputOpts = dim100_Input()
% % dim100_Input - 100D Gaussian inputs (matched mean/variance to the original uniforms)
% %
% % Original:
% %   ii ~= 20 : Uniform[1,2]
% %   ii == 20 : Uniform[1,3]
% %
% % Here we use Normal(mu, sigma) with the same mean and variance:
% %   Uniform[a,b]: mu = (a+b)/2,  sigma = (b-a)/sqrt(12)
% 
% for ii = 1:100
%     InputOpts.Marginals(ii).Type = 'Gaussian';
% 
%     if ii == 20
%         a = 1; b = 3;
%     else
%         a = 1; b = 2;
%     end
% 
%     mu    = 0.5*(a + b);
%     sigma = (b - a) / sqrt(12);
% 
%     % UQLab convention: Parameters = [mean std]
%     InputOpts.Marginals(ii).Parameters = [mu sigma];
% end
% 
% end
