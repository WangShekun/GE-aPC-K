function y_pred = aPC_PCKPredictBlock(Psi_new, U_new, PCKModel)

    trend_idx = PCKModel.trend_idx;
    Fnew = Psi_new(:, trend_idx);
    U_train = PCKModel.U_train;
    theta = PCKModel.theta;
    Kff_new = pckCorrMatrix(U_new, U_train, theta, PCKModel.corr);
    mu0 = Fnew * PCKModel.beta + Kff_new * PCKModel.alpha;
    [~, Kfg_blocks] = pckEvalKernel(U_new, U_train, theta, PCKModel.corr);

    blocks = PCKModel.blocks;
    B = numel(blocks);
    weights = localAllBlockScores(U_new, blocks, PCKModel.tau_dist);
    corrections = zeros(size(U_new, 1), B);
    for bidx = 1:B
        block = blocks{bidx};
        kxg = localCrossBlockFromBlocks(Kfg_blocks, block.sample_idx);
        corrections(:, bidx) = (kxg - Kff_new * block.C) * block.eta;
    end

    y_pred = mu0 + sum(weights .* corrections, 2);
end

function scores = localAllBlockScores(U_new, blocks, tau_dist)
    B = numel(blocks);
    scores = zeros(size(U_new, 1), B);
    for bidx = 1:B
        delta = U_new - blocks{bidx}.center;
        scores(:, bidx) = exp(-sum(delta .^ 2, 2) / max(tau_dist ^ 2, eps));
    end
    scores = scores ./ max(sum(scores, 2), eps);
end

function kxg = localCrossBlockFromBlocks(Kfg_blocks, sample_idx)
    d = numel(Kfg_blocks);
    cells = cell(1, d);
    for a = 1:d
        cells{a} = Kfg_blocks{a}(:, sample_idx);
    end
    kxg = cat(2, cells{:});
end
