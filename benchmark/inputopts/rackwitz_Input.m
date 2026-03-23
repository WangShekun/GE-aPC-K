function InputOpts = rackwitz_Input()
% Input definition for Rackwitz high-dimensional reliability problem
%
% ---- CHANGE DIMENSION HERE ----
d = 200;    % e.g. 40 / 100 / 150 / 200 / 250 / 300

mu    = 1.0;
sigma = 0.2;

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Lognormal';
    InputOpts.Marginals(ii).Parameters = [mu sigma];
end
end
