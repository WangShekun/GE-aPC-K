function InputOpts = wingweight_Input()
InputOpts.Marginals(1).Type = 'Uniform';
InputOpts.Marginals(1).Parameters = [150 200];
InputOpts.Marginals(2).Type = 'Uniform';
InputOpts.Marginals(2).Parameters = [220 300];
InputOpts.Marginals(3).Type = 'Uniform';
InputOpts.Marginals(3).Parameters = [6 10];
InputOpts.Marginals(4).Type = 'Uniform';
InputOpts.Marginals(4).Parameters = [-10 10];
InputOpts.Marginals(5).Type = 'Uniform';
InputOpts.Marginals(5).Parameters = [16 45];
InputOpts.Marginals(6).Type = 'Uniform';
InputOpts.Marginals(6).Parameters = [0.5 1];
InputOpts.Marginals(7).Type = 'Uniform';
InputOpts.Marginals(7).Parameters = [0.08 0.18];
InputOpts.Marginals(8).Type = 'Uniform';
InputOpts.Marginals(8).Parameters = [2.5 6];
InputOpts.Marginals(9).Type = 'Uniform';
InputOpts.Marginals(9).Parameters = [1700 2500];
InputOpts.Marginals(10).Type = 'Uniform';
InputOpts.Marginals(10).Parameters = [0.025 0.08];
end

% function InputOpts = wingweight_Input()
% % wingweight_Input - Gaussian version
% % Each variable: Normal(mu, sigma^2) with
% %   mu    = (a+b)/2
% %   sigma = (b-a)/6   (treat original [a,b] as approx ±3σ)
% 
% ranges = [ ...
%     150   200;     % 1
%     220   300;     % 2
%     6     10;      % 3
%     -10   10;      % 4
%     16    45;      % 5
%     0.5   1;       % 6
%     0.08  0.18;    % 7
%     2.5   6;       % 8
%     1700  2500;    % 9
%     0.025 0.08];   % 10
% 
% for i = 1:size(ranges,1)
%     a = ranges(i,1);
%     b = ranges(i,2);
% 
%     mu = 0.5*(a+b);
%     sigma = (b-a)/6;
% 
%     InputOpts.Marginals(i).Type = 'Gaussian';          % UQLab: Normal
%     InputOpts.Marginals(i).Parameters = [mu sigma];    % [mean, std]
% end
% 
% end
