function InputOpts = ishigamiHD_Input()
d = 200;   % <<< 改这里

for ii = 1:d
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [-pi pi];
end
end
