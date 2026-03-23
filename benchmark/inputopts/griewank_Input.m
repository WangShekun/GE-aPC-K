function InputOpts = griewank_Input()
% Input definition for Griewank function
%
% ---- CHANGE DIMENSION HERE ----
d = 200;   % e.g. 30, 60, 120, 180, 240, 320
% --------------------------------

lb = -10;
ub =  10;

for ii = 1:d
    InputOpts.Marginals(ii).Type       = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [lb ub];
end
end
