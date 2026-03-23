function InputOpts = dimN_Input(N)
% -------------------------------------------------------------------------
% Input definition for the paper analytic problem
% Xi ~ U(1,2), except X20 ~ U(1,3)
% -------------------------------------------------------------------------

if nargin < 1
    N = 200;
end

for ii = 1:N
    InputOpts.Marginals(ii).Type = 'Uniform';
    if ii == 20
        InputOpts.Marginals(ii).Parameters = [1 3];
    else
        InputOpts.Marginals(ii).Parameters = [1 2];
    end
end

end
