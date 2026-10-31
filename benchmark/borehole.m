function [Y] = borehole(X)
% borehole - 8D groundwater flow test function
% 
% Standard borehole function from:
%   Morris, M.D. (1991). Factorial sampling plans for preliminary 
%   computational experiments. Technometrics, 33(2), 161-174.
%
% Inputs (8 variables):
%   X(:,1): rw - radius of borehole (m), typically [0.05, 0.15]
%   X(:,2): r  - radius of influence (m), typically [100, 50000]
%   X(:,3): Tu - transmissivity of upper aquifer (m^2/yr), typically [63070, 115600]
%   X(:,4): Hu - potentiometric head of upper aquifer (m), typically [990, 1110]
%   X(:,5): Tl - transmissivity of lower aquifer (m^2/yr), typically [63.1, 116]
%   X(:,6): Hl - potentiometric head of lower aquifer (m), typically [700, 820]
%   X(:,7): L  - length of borehole (m), typically [1120, 1680]
%   X(:,8): Kw - hydraulic conductivity of borehole (m/yr), typically [9855, 12045]
%
% Output:
%   Y - flow rate through the borehole (m^3/yr)

    rw = X(:,1);
    r = X(:,2);
    Tu = X(:,3);
    Hu = X(:,4);
    Tl = X(:,5);
    Hl = X(:,6);
    L = X(:,7);
    Kw = X(:,8);
    
    % Numerator
    numerator = 2 * pi .* Tu .* (Hu - Hl);
    
    % Denominator (log term)
    log_term = log(r ./ rw);
    
    % First part of denominator
    denom1 = (2 .* L .* Tu) ./ (log_term .* rw.^2 .* Kw);
    
    % Second part of denominator  
    denom2 = Tu ./ Tl;
    
    % Total denominator
    denominator = log_term .* (1 + denom1 + denom2);
    
    % Flow rate
    Y = numerator ./ denominator;
end
