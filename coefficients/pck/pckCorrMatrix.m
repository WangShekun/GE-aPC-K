function R = pckCorrMatrix(X1, X2, theta, corr_type)
    % Correlation matrix interface shared by APCK training and prediction.
    R = pckEvalKernel(X1, X2, theta, corr_type);

    if isequal(size(X1), size(X2)) && isequal(X1, X2)
        R = (R + R') * 0.5;
        R(1:size(R,1)+1:end) = 1;
    end
end
