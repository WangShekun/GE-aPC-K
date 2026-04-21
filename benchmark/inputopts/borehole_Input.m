function InputOpts = borehole_Input()
% borehole_Input - Input distribution for borehole function
%
% Standard uniform distributions for borehole problem:
%   rw: Uniform[0.05, 0.15] - radius of borehole (m)
%   r:  Uniform[100, 50000] - radius of influence (m)
%   Tu: Uniform[63070, 115600] - transmissivity of upper aquifer (m^2/yr)
%   Hu: Uniform[990, 1110] - potentiometric head of upper aquifer (m)
%   Tl: Uniform[63.1, 116] - transmissivity of lower aquifer (m^2/yr)
%   Hl: Uniform[700, 820] - potentiometric head of lower aquifer (m)
%   L:  Uniform[1120, 1680] - length of borehole (m)
%   Kw: Uniform[9855, 12045] - hydraulic conductivity of borehole (m/yr)

    % Variable 1: rw
    InputOpts.Marginals(1).Type = 'Uniform';
    InputOpts.Marginals(1).Parameters = [0.05, 0.15];
    
    % Variable 2: r
    InputOpts.Marginals(2).Type = 'Uniform';
    InputOpts.Marginals(2).Parameters = [100, 50000];
    
    % Variable 3: Tu
    InputOpts.Marginals(3).Type = 'Uniform';
    InputOpts.Marginals(3).Parameters = [63070, 115600];
    
    % Variable 4: Hu
    InputOpts.Marginals(4).Type = 'Uniform';
    InputOpts.Marginals(4).Parameters = [990, 1110];
    
    % Variable 5: Tl
    InputOpts.Marginals(5).Type = 'Uniform';
    InputOpts.Marginals(5).Parameters = [63.1, 116];
    
    % Variable 6: Hl
    InputOpts.Marginals(6).Type = 'Uniform';
    InputOpts.Marginals(6).Parameters = [700, 820];
    
    % Variable 7: L
    InputOpts.Marginals(7).Type = 'Uniform';
    InputOpts.Marginals(7).Parameters = [1120, 1680];
    
    % Variable 8: Kw
    InputOpts.Marginals(8).Type = 'Uniform';
    InputOpts.Marginals(8).Parameters = [9855, 12045];
end
