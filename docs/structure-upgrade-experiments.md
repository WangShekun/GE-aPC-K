# Structure Upgrade Experiments

This file records the main structural experiments attempted after identifying that `geapck_block` often behaves like `APCK backbone + local gradient correction` and does not reliably approach full GEAPCK accuracy.

## 1. GEAPCK_BLOCK Backbone LOO Upgrade

- Goal: make the `geapck_block` model-selection score more faithful to the backbone Kriging model by replacing the simplified `diag(Kff^{-1})` denominator with the Universal Kriging projected-precision diagonal.
- Files touched:
  - `coefficients/aPC_PCKBackendBlock.m`
  - `tests/TestModelTypes.m`
- Change:
  - `results.LOO` for `geapck_block` now uses a UK-style backbone LOO.
  - The previous simplified score is still stored as `results.PCKModel.backbone_legacy_loo` for diagnostics.
- Evidence:
  - Unit tests pass for the new diagnostics.
  - `tmp/oned_compare_block.m` showed that the new LOO was more sensitive to different block configurations.
- Outcome:
  - Partially useful.
  - This made the score more reasonable, but it did **not** materially improve validation `RelMSE` by itself.
- Status:
  - Kept.

## 2. APCK Gradient-Aware Theta Proxy

- Goal: move gradient information into APCK theta optimization without paying the full GEAPCK augmented-system cost.
- Files touched during the experiment:
  - `apc_train/aPC_TrainCandidate.m`
  - `coefficients/aPC_PCKBackend.m`
  - `tests/TestModelTypes.m`
  - `tmp/wingweight_apck_gradient_proxy_compare.m`
- Prototype:
  - Add an optional penalty to the APCK theta objective using the mismatch between trend-implied gradients `dF*beta` and model-space gradients `G`.
- Benchmark used:
  - `wingweight`, with PLS-4 reduction and the same train/validation split for baseline and prototype.
- Observed result:
  - Baseline APCK: `RelMSE = 1.320261e-02`
  - APCK + gradient-theta proxy: `RelMSE = 1.320261e-02`
  - Selected theta remained at the same lower-bound solution in both runs.
- Interpretation:
  - The proxy path executed correctly, but it did not move the optimizer to a different solution and did not improve accuracy.
- Outcome:
  - Ineffective.
- Status:
  - Reverted.

## 3. APCK Gradient-Aware Trend Selection Score

- Goal: instead of changing theta optimization, use gradient consistency to influence APCK trend selection.
- Files touched during the experiment:
  - `apc_train/aPC_TrainCandidate.m`
  - `coefficients/aPC_PCKBackend.m`
  - `tests/TestModelTypes.m`
  - `tmp/wingweight_apck_gradient_trend_compare.m`
- Prototype:
  - Add a trend-selection score of the form `LOO + lambda * normalized_gradient_mismatch`, where the gradient mismatch was again based on `dF*beta` versus model-space gradients.
- Benchmark used:
  - `wingweight`, same split and baseline as above.
- Observed result:
  - Baseline APCK: `RelMSE = 1.320261e-02`
  - APCK + gradient-trend score: `RelMSE = 2.709489e-02`
  - The prototype selected a smaller trend (`TrendK = 3` instead of `5`) and degraded accuracy.
- Interpretation:
  - The added gradient term dominated the selection score and pushed the model toward a worse function-value predictor.
- Outcome:
  - Harmful.
- Status:
  - Reverted.

## Current Takeaway

- Improving the **backbone LOO formula** was worth keeping for `geapck_block`, but it was not enough to raise final predictive accuracy.
- Two simple attempts to move gradients into the APCK backbone using only `dF*beta`-based mismatch were not successful:
  - one had no measurable effect,
  - one made the model worse.
- This suggests that if gradient information is to be moved into the main structure, a stronger mechanism is needed than a trend-only mismatch term.

## Recommended Next Directions

The next structural experiments should focus on mechanisms that preserve more of the joint value-gradient coupling, for example:

1. A stronger gradient proxy that includes kernel-residual gradient information, not just `dF*beta`.
2. Partial global coupling, such as a reduced augmented system on a selected subset of gradient observations.
3. Residualized local correction with explicit shrinkage, so the correction cannot overpower a good APCK backbone.

## 4. Sparse Global GE Backbone + Residualized Local Correction

- Goal: upgrade `geapck_block` from `APCK backbone + local gradient correction` to `sparse GE backbone + residualized local correction`.
- Files touched:
  - `coefficients/aPC_PCKBackendBlock.m`
  - `coefficients/pck/pckBuildAugmentedSystem.m`
  - `predict/aPC_PCKPredictBlock.m`
  - `tests/TestModelTypes.m`
  - `tmp/compare_sparse_ge_block_v1.m`
- Prototype fields:
  - `GEBlockOpts.global_grad_ratio`
  - `GEBlockOpts.global_grad_select`
  - `GEBlockOpts.num_blocks`
- Prototype design:
  - Select a global gradient subset by maximin sampling.
  - Build a reduced augmented backbone with all function values plus only the selected gradients.
  - Predict backbone gradients on all training points.
  - Form residual gradients and fit local block corrections only to those residuals.
- Benchmarks used:
  - `wingweight`
  - `ishigami`
- Observed result:
  - `global_grad_ratio = 0.0` behaves like the previous block baseline and remains stable.
  - `global_grad_ratio = 0.2` is harmful.
  - On `wingweight`, `RelMSE` deteriorated from about `1.320261e-02` to about `5.657357e+07`.
  - On `ishigami`, `RelMSE` deteriorated from about `3.580251e-06` to about `8.144403e-02`.
- Interpretation:
  - This first sparse-coupling version destabilized the model instead of improving it.
  - The backbone-to-residual handoff is not yet trustworthy enough for default use.
- Outcome:
  - Harmful in v1.
- Status:
  - Kept only as an explicit experimental path.
  - Default `GEBlockOpts.global_grad_ratio` is set to `0.0`, so the main behavior does not change unless the user explicitly enables the prototype.

## 5. GEAPCK Gradient Scaling Experiment

- Goal: test whether rescaling the gradient block in the augmented system could improve conditioning and accuracy.
- Files touched:
  - `coefficients/pck/pckBuildAugmentedSystem.m`
  - `tests/TestModelTypes.m`
- Prototype design:
  - Add a `GEScaling` field to control gradient block scaling.
  - Test multipliers: 0.1, 1.0 (baseline), 10.0.
- Benchmarks used:
  - `wingweight`
  - `ishigami`
- Observed result:
  - `wingweight`: scaling changes had minimal impact on final RelMSE; theta optimization compensated for the scaling.
  - `ishigami`: similar behavior—scaling did not materially change the outcome.
- Interpretation:
  - The augmented system already balances value and gradient contributions appropriately through the kernel structure.
  - Manual scaling does not provide a reliable accuracy boost.
- Outcome:
  - Ineffective.
- Status:
  - Reverted. The prototype code is not kept in the main branch.

## 6. Low-Rank / Modal GE Prototype

- Goal: compress gradient information itself, rather than sampling only a subset of gradient points or adding post-hoc block corrections.
- Files touched:
  - `apc_train/aPC_TrainCandidate.m`
  - `apc_train/aPC_Train.m`
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `coefficients/pck/pckBuildLowRankAugmentedSystem.m`
  - `predict/aPC_Predict.m`
  - `predict/aPC_PCKPredictLowRank.m`
  - `tests/TestModelTypes.m`
  - `tmp/compare_ge_lowrank_v1.m`
- Prototype design:
  - New model type: `geapck_lowrank`.
  - Work in the already reduced/model space.
  - Compute an uncentered thin SVD of the model-space gradient matrix `G`.
  - Keep the first `r` right singular vectors and replace raw directional derivatives by directional-derivative modes `Z = G * V_r`.
  - Build an augmented GE system on `[Y; vec(Z)]` and the projected gradient trend matrix.
- Benchmarks used:
  - `ishigami`, `Order=12`, `q=0.8`, `N=300`
  - `wingweight`, `PLS-4`, `Order=2`, `q=0.8`, `N=80`
- Observed result:
  - `ishigami`:
    - `GEAPCK`: `RelMSE ≈ 1.659e-12`, `Train ≈ 38.9s`
    - `GEAPCK_BLOCK`: `RelMSE ≈ 3.339e-12`, `Train ≈ 28.8s`
    - `GEAPCK_LOWRANK(r=2)`: `RelMSE ≈ 1.231e-12`, `Train ≈ 26.7s`
    - `GEAPCK_LOWRANK(r=3)`: `RelMSE ≈ 1.473e-12`, `Train ≈ 47.7s`
  - `wingweight` (PLS-4 reduction):
    - `APCK`: `RelMSE ≈ 1.230e-02`
    - `GEAPCK`: `RelMSE ≈ 4.253e-02`
    - `GEAPCK_BLOCK`: `RelMSE ≈ 1.230e-02`
    - `GEAPCK_LOWRANK(r=2)`: `RelMSE ≈ 3.147e-01`
    - `GEAPCK_LOWRANK(r=3)`: `RelMSE ≈ 1.497e-02`
  - `wingweight` (no reduction, p=2, q=0.8, N=80):
    - `GEAPCK`: `RelMSE ≈ 6.92e-07`
    - `GEAPCK_BLOCK`: `RelMSE ≈ 6.10e-05`
    - `GEAPCK_LOWRANK(r=3)`: `RelMSE ≈ 9.03e-06`
- Interpretation:
  - This is the first new direction that showed a meaningful positive signal on at least one representative case.
  - After upgrading the implementation to direct projected-kernel assembly, the low-rank path no longer materializes full derivative blocks before projection in either training or prediction.
  - On `ishigami`, both `r=2` and `r=3` remain competitive with full GE; in this run `r=2` was the best low-rank setting.
  - On `wingweight` with PLS-4, `r=3` still improves greatly over full `GEAPCK` but still does not beat `APCK`/`GEAPCK_BLOCK`; `r=2` remains too aggressive.
  - On `wingweight` without dimension reduction, low-rank GE (`r=3`) significantly outperformed block GE and approached the accuracy of full GE, though still one order of magnitude worse than full GE.
  - The projected-kernel refactor removes the main structural inefficiency of the original prototype; future work should focus on when low-rank GE helps most, rather than on the old full-block assembly bottleneck.
- Outcome:
  - Promising, and now implemented with direct projected assembly.
- Status:
  - Keep as an experimental branch for further development and broader benchmark testing.
  - The next step, if continued, should focus on rank-selection strategy and broader benchmark coverage rather than kernel-assembly mechanics.

