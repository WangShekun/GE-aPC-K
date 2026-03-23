function aPC(varargin)
% aPC — Initialize the PolyChaos framework
%
% This script registers all major modules and their subdirectories
% into MATLAB's search path. The structure is flat: "core" and
% all module directories reside at the same hierarchy level.
%
% Usage:
%   aPC          % initialize paths
%   aPC reset    % clear old paths and reinitialize

    %--------------------------------------------------------------
    % 1. Determine the root directory
    %--------------------------------------------------------------
    core_dir = fileparts(mfilename('fullpath'));    % …/core
    root_dir = fileparts(core_dir);                % …/

    %--------------------------------------------------------------
    % 2. Define all modules at the same level as 'core'
    %--------------------------------------------------------------
    modules = {
        'basis'
        'coefficients'
        'display'
        'utils'
        'predict'
        'UQ'
        'sensitivity'
        'dimreduct'
        'benchmark'
        'apc_train'
        'auto_search'
    };

    %--------------------------------------------------------------
    % 3. Handle optional 'reset'
    %--------------------------------------------------------------
    if nargin > 0 && strcmpi(varargin{1}, 'reset')
        aPC_remove_paths(root_dir, modules);
    end

    %--------------------------------------------------------------
    % 4. Add all modules recursively
    %--------------------------------------------------------------
    for k = 1:numel(modules)
        module_path = fullfile(root_dir, modules{k});
        if exist(module_path, 'dir')
            addpath(genpath(module_path));
        else
            warning('[aPC] Module not found: %s', modules{k});
        end
    end

    % 5. Add the core folder itself
    addpath(core_dir);

    fprintf('[aPC] Initialization successful. Root = %s\n', root_dir);
end

% -------------------------------------------------------------------------
function aPC_remove_paths(root_dir, modules)
% Remove previously registered module paths to avoid duplicates

    for k = 1:numel(modules)
        mp = fullfile(root_dir, modules{k});
        if exist(mp, 'dir')
            rmpath(genpath(mp));
        end
    end
end
