function InputOpts = dim300_pce_Input()
% -------------------------------------------------------------------------
% Input for dim300_pce (All Gaussian) — best for Hermite PC / aPC
% Editable dimension here:
d = 300;
% -------------------------------------------------------------------------

mu = 0;
sig = 1;

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Gaussian';
    InputOpts.Marginals(ii).Parameters = [mu sig];
end
end

% function InputOpts = dim300_pce_Input()
% % -------------------------------------------------------------------------
% % Input for dim300_pce (Mixed marginals, PCE-friendly)
% % Editable dimension here:
% d = 300;
% % -------------------------------------------------------------------------
% 
% for ii = 1:d
%     if ii <= 20
%         % bounded, moderate range
%         InputOpts.Marginals(ii).Type       = 'Uniform';
%         InputOpts.Marginals(ii).Parameters = [-1 1];
% 
%     elseif ii <= 60
%         % positive, mild variability (keep sigma small to avoid heavy tails)
%         InputOpts.Marginals(ii).Type       = 'Lognormal';
%         InputOpts.Marginals(ii).Parameters = [1 0.2];
% 
%     else
%         % centered-ish, moderate variance
%         InputOpts.Marginals(ii).Type       = 'Gaussian';
%         InputOpts.Marginals(ii).Parameters = [0 1];
%     end
% end
% end

