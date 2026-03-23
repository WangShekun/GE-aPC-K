classdef SupervisedKernelPCAOption < handle

    methods(Static)

        % ==============================
        %        TRAIN PRINTER
        % ==============================
        function displayTrain(skpca)
            fprintf('\n');
            fprintf('*** SKPCA (HSIC) model training finished ***\n');
            fprintf('running time            = %.4f seconds\n', skpca.runningTime);
            fprintf('kernel function (X)     = %s \n', skpca.kernelFunc.type);
            fprintf('kernel function (Y)     = %s \n', skpca.kernelY.type);
            fprintf('number of samples       = %d \n', skpca.numSamples);
            fprintf('number of components    = %d \n', skpca.numComponents);
        end


        % ==============================
        %        TEST PRINTER
        % ==============================
        function displayTest(result)
            fprintf('\n');
            fprintf('*** SKPCA (HSIC) model test finished ***\n');
            fprintf('running time            = %.4f seconds\n', result.runningTime);
            fprintf('number of test data     = %d \n', size(result.score,1));
        end

    end
end
