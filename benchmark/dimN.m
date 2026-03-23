function [Y] = dimN(X)
% -------------------------------------------------------------------------
% Paper analytic problem (Eq. 28-29): scalable to arbitrary N
%
% Input:
%   X : Ns × N   (each row is one sample)
%
% Output:
%   Y : Ns × 1
% -------------------------------------------------------------------------

[Ns, N] = size(X);

% ----- summed terms -----
S1 = sum(X, 2);                                    % Ns×1
S3 = sum(X.^3, 2);                                 % Ns×1
Slog = sum(log(X.^2 + X.^4), 2);                   % Ns×1

Y = 3 ...
    - (5/N)  * S1 ...
    + (1/N)  * S3 ...
    + (1/(3*N)) * Slog;

% ----- low-index interaction terms (require N>=5 to exist) -----
if N >= 5
    Y = Y + X(:,1).*X(:,2).^2 + X(:,2).*X(:,4) - X(:,3).*X(:,5);
else
    % 如果你真要跑 N<5（论文没有），这里给出最小保护
    if N >= 2
        Y = Y + X(:,1).*X(:,2).^2;
    end
    if N >= 4
        Y = Y + X(:,2).*X(:,4);
    end
    if N >= 5
        Y = Y - X(:,3).*X(:,5);
    end
end

% ----- indicator-gated high-index terms -----
if N >= 51
    Y = Y + X(:,51);
end

if N >= 54
    % I50*I54 automatically satisfied when N>=54
    Y = Y + X(:,50) .* (X(:,54).^2);
elseif N >= 50
    % N in [50,53]: I50=1 but I54=0 -> term vanishes
    % do nothing
end

end
