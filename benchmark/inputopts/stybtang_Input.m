function InputOpts = stybtang_Input()
% Input definition for Styblinski–Tang function
%
% ---- CHANGE DIMENSION HERE ----
d = 150;   % e.g. 60 / 80 / 150 / 200

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-5 5];
end
end
