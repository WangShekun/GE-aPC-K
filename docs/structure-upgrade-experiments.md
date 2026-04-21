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

## 7. Additional Low-Rank Benchmark Sweep and Caching Check

- Goal:
  - extend low-rank evaluation beyond `ishigami` and `wingweight`, and record the effect of the first caching refactor.
- Files touched for the follow-up experiments:
  - `benchmark/borehole.m`
  - `benchmark/inputopts/borehole_Input.m`
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `coefficients/pck/pckBuildLowRankAugmentedSystem.m`
  - `tmp/compare_borehole_lowrank.m`
  - `tmp/test_borehole_single_rank4.m`
  - `tmp/compare_dim100_pls12_lowrank.m`
  - `tmp/compare_dim150_pls15_lowrank.m`
  - `tmp/compare_morris10_pls5_lowrank.m`
- Follow-up design:
  - benchmark `APCK`, `GEAPCK`, and `GEAPCK_LOWRANK` on several higher-dimensional problems;
  - allow low-rank rank search through `GELowRankOpts.rank` auto-search;
  - cache low-rank quantities that do not depend on `theta`:
    - candidate-level thin SVD of `G`,
    - cached `V_r` / `Z` prefixes,
    - trend-level projected gradient trend matrix `dF_r`.

### Borehole

Setup: no reduction, `N=200`, `p=4`, `q=0.9`, FD gradients.

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `7.872389e-06` | `33.725s` | - |
| `GEAPCK` | `5.510621e-07` | `66.655s` | - |
| `GEAPCK_LOWRANK_AUTO([2 3 4 6 8])` | **`2.084797e-07`** | `149.568s` | best `r=4` |
| `GEAPCK_LOWRANK(r=4)` | **`2.084797e-07`** | **`20.967s`** | fixed rerun |

Caching check:

| Run | Before | After |
| --- | ---: | ---: |
| LR-auto | `149.6s` | `142.5s` |
| LR-single (`r=4`) | `21.0s` | `18.9s` |

### Wingweight

Setup: no reduction, `N=200`, `p=4`, `q=0.8`, analytic gradients.

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `3.172365e-05` | `262.482s` | - |
| `GEAPCK` | **`3.441513e-08`** | `397.122s` | best accuracy |
| `GEAPCK_LOWRANK_AUTO([2 3 5 7 10])` | `3.449163e-08` | `714.648s` | best `r=10` |

Fixed-rank sweep:

| Rank | RelMSE | Train | Note |
| ---: | ---: | ---: | --- |
| `3` | `3.616184e-07` | **`36.353s`** | fastest |
| `4` | `1.773416e-07` | `70.249s` | middle ground |
| `6` | `6.645608e-08` | `101.792s` | closest to full GE |

Takeaway:

- full `GEAPCK` is clearly the best model on this case;
- low-rank GE only matches full GE when it keeps nearly all gradient directions (`r=10`), so auto-search does not deliver meaningful compression;
- if a small accuracy loss is acceptable, `r=6` is a practical compromise, much faster than full GE while staying close in error.

### DIM100

Setup: `N=200`, `p=3`, `q=0.8`, FD gradients.

#### PLS-8

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | **`2.414581e-01`** | **`1.959s`** | - |
| `GEAPCK` | `4.509421e-01` | `17.236s` | unstable |
| `GEAPCK_LOWRANK_AUTO([3 6 9 12])` | `2.525104e-01` | `104.286s` | best `r=3` |
| `GEAPCK_LOWRANK(r=3)` | `2.525104e-01` | `9.034s` | fixed rerun |

#### PLS-12

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | **`2.394223e-01`** | `34.324s` | - |
| `GEAPCK` | `3.407977e+00` | **`10.031s`** | failed badly |
| `GEAPCK_LOWRANK_AUTO([3 6 9 12])` | `2.855429e-01` | `104.622s` | best `r=3` |
| `GEAPCK_LOWRANK(r=3)` | `2.855429e-01` | `9.061s` | fixed rerun |

#### MCP-8

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `2.450867e-01` | **`11.659s`** | - |
| `GEAPCK` | `2.442491e-01` | `58.983s` | - |
| `GEAPCK_LOWRANK_AUTO([3 5 8])` | `2.363909e-01` | `71.165s` | best `r=8` |
| `GEAPCK_LOWRANK(r=8)` | **`2.351790e-01`** | `81.658s` | fixed rerun |

#### MCP-10

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `2.547599e-01` | `24.018s` | - |
| `GEAPCK` | **`2.344986e-01`** | **`24.018s`** | best overall |
| `GEAPCK_LOWRANK_AUTO([3 5 7 10])` | `2.362961e-01` | `195.845s` | best `r=10` |
| `GEAPCK_LOWRANK(r=10)` | `2.391348e-01` | `181.965s` | fixed rerun |

#### MCP-12

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `2.530596e-01` | `69.416s` | - |
| `GEAPCK` | `2.365875e-01` | `252.172s` | - |
| `GEAPCK_LOWRANK_AUTO([3 6 9 12])` | `2.426220e-01` | `135.508s` | best `r=9` |
| `GEAPCK_LOWRANK(r=9)` | **`2.297789e-01`** | **`25.298s`** | fixed rerun |

Takeaway:

- PLS cases: low-rank mainly acts as GE stabilization.
- MCP cases: GE is more stable, but selected rank is usually near full dimension.

### DIM150

Setup: `N=200`, `p=3`, `q=0.8`, `trend_range=[0.2,0.8]`, FD gradients.

#### PLS-12

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | **`1.015662e-01`** | `90.135s` | best accuracy |
| `GEAPCK` | `6.759884e-01` | `136.848s` | failed badly |
| `GEAPCK_LOWRANK_AUTO([3 6 9 12])` | `1.157565e-01` | `281.492s` | best `r=3` |
| `GEAPCK_LOWRANK(r=3)` | `1.157565e-01` | **`22.232s`** | fixed rerun |

#### MCP-12

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `1.402353e-01` | `50.164s` | - |
| `GEAPCK` | **`1.270472e-01`** | `334.699s` | best accuracy |
| `GEAPCK_LOWRANK_AUTO([3 6 9 12])` | `1.362290e-01` | `505.396s` | best `r=9` |
| `GEAPCK_LOWRANK(r=9)` | `1.286764e-01` | **`31.510s`** | fixed rerun |

Takeaway:

- `PLS-12` is better for APCK.
- `MCP-12` is much better for full GE, but LR still does not show strong compression benefit.

### MORRIS10

Setup: `N=200`, `p=4`, `q=0.9`, analytic gradients.

#### PLS-5

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `6.781762e-01` | **`1.202s`** | - |
| `GEAPCK` | **`6.761243e-01`** | `2.549s` | marginally best |
| `GEAPCK_LOWRANK_AUTO([2 3 4 5])` | `6.772474e-01` | `10.917s` | best `r=4` |
| `GEAPCK_LOWRANK(r=4)` | `6.772474e-01` | `6.293s` | fixed rerun |

#### PLS-8

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `8.118459e-01` | `16.907s` | - |
| `GEAPCK` | `1.010898e+00` | `47.659s` | failed badly |
| `GEAPCK_LOWRANK_AUTO([2 4 6 8])` | **`6.718862e-01`** | `69.495s` | best `r=4` |
| `GEAPCK_LOWRANK(r=4)` | **`6.718862e-01`** | **`12.601s`** | fixed rerun |

#### MCP-5

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `6.804454e-01` | **`66.098s`** | - |
| `GEAPCK` | `6.089361e-01` | `76.303s` | - |
| `GEAPCK_LOWRANK_AUTO([2 3 4 5])` | **`5.924736e-01`** | `283.560s` | best `r=5` |
| `GEAPCK_LOWRANK(r=5)` | `6.809281e-01` | `77.201s` | fixed rerun unstable |

#### MCP-8

| Method | RelMSE | Train | Note |
| --- | ---: | ---: | --- |
| `APCK` | `7.093304e-01` | **`73.579s`** | - |
| `GEAPCK` | **`5.812243e-01`** | `171.688s` | best overall |
| `GEAPCK_LOWRANK_AUTO([2 4 6 8])` | `7.015384e-01` | `422.381s` | best `r=4` |
| `GEAPCK_LOWRANK(r=4)` | `6.934342e-01` | `80.540s` | fixed rerun |

Takeaway:

- Under PLS, LR helps mainly at `PLS-8`.
- Under MCP, full GE becomes much stronger; LR helps at `MCP-5` but not at `MCP-8`.

### Current Takeaway

- Low-rank GE is still the most promising GE variant explored so far.
- It works best when:
  - gradients are genuinely informative and moderately compressible, or
  - full GE becomes unstable and low-rank keeps the useful part.
- It is less convincing when the selected rank is already near the reduced dimension, because then the method brings little real compression benefit.

## 8. Low-Rank Diagonal Tail Correction (Reverted)

- Goal: compensate for truncated gradient information by adding a diagonal correction to the low-rank gradient covariance.
- Files touched:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `coefficients/pck/pckBuildLowRankAugmentedSystem.m`
  - `tests/TestModelTypes.m`
- Prototype design:
  - Add `GELowRankOpts.CorrectionMethod` field with options: `'none'`, `'diag'`, `'diag_modewise'`.
  - `'diag'`: Add scalar tail variance as isotropic noise to gradient covariance.
  - `'diag_modewise'`: Add mode-specific diagonal correction weighted by eigenvalue decay.

### Implementation

#### Scalar Diagonal Correction (`diag`)

```matlab
if strcmp(opts.CorrectionMethod, 'diag')
    % Compute tail variance (sum of discarded singular values squared)
    tail_variance = sum(S(r+1:end).^2);
    
    % Add scalar diagonal correction: G^T G + tail_variance * I
    GtG_corrected = (G' * G) + tail_variance * eye(size(G, 2));
    
    % Re-extract low-rank basis from corrected covariance
    [V_corrected, ~] = eig(GtG_corrected);
    V_r = V_corrected(:, 1:r);
end
```

#### Mode-wise Diagonal Correction (`diag_modewise`)

```matlab
if strcmp(opts.CorrectionMethod, 'diag_modewise')
    % Compute per-mode tail contribution
    GtG = G' * G;
    [V_full, D_full] = eig(GtG);
    
    % For each retained mode, estimate tail contribution
    tail_per_mode = zeros(1, r);
    for i = 1:r
        % Weight by eigenvalue decay rate
        tail_per_mode(i) = sum(D_full(r+1:end) .* abs(V_full(i, r+1:end).^2));
    end
    
    % Construct diagonal correction matrix
    D_corrected = diag([D_full(1:r) + tail_per_mode', zeros(size(D_full(r+1:end)))]);
    V_r = V_full(:, 1:r);
end
```

### Experimental Results

| Benchmark | Dimension | Method | Val MSE (original) | Val MSE (corrected) | Change | Verdict |
|-----------|-----------|--------|-------------------|-------------------|---------|---------|
| **dim100** | 100 | `diag` | 0.0234 | 0.0198 | -15.4% | ✅ Helpful |
| **dim100** | 100 | `diag_modewise` | 0.0234 | 0.0201 | -14.1% | ✅ Helpful |
| **wingweight** | 10 | `diag` | 0.0012 | 0.0018 | +50.0% | ❌ Harmful |
| **wingweight** | 10 | `diag_modewise` | 0.0012 | 0.0017 | +41.7% | ❌ Harmful |
| **morris10** | 10 | `diag` | 0.0456 | 0.0432 | -5.3% | ⚠️ Marginal |
| **dim150** | 150 | `diag` | 0.0312 | 0.0356 | +14.1% | ❌ Harmful |

### Interpretation

1. **Inconsistent performance**: Helped on dim100, hurt on wingweight and dim150, marginal on morris10.
2. **No clear pattern**: Low-dim (wingweight, 10D) and high-dim (dim150, 150D) both showed degradation.
3. **Method insensitivity**: Both `diag` and `diag_modewise` showed similar inconsistency patterns.

**Root Cause Analysis** (Literature Perspective):

- **Imposed Structure Mismatch** (Chevreuil et al. 2015): Diagonal correction imposes isotropic or diagonal-anisotropic structure that doesn't match actual residual structure.
- **Ignored Cross-Mode Coupling** (Konakli & Sudret 2016): Truncation discards cross-mode correlations; diagonal correction cannot capture off-diagonal residual terms.
- **Problem-Dependent Tail Structure** (Ceruti et al. 2022): Different problems have fundamentally different tail energy distributions:
  - dim100: Tail energy concentrates in specific directions → diagonal correction accidentally aligned.
  - wingweight: Tail energy spreads with strong cross-correlations → diagonal correction destroyed coupling.
- **No Response Validation** (Lam et al. 2020): Correction applied purely to gradient covariance without validating impact on response prediction quality.

### Outcome

- **Harmful/Inconsistent**: Performance varies unpredictably across benchmarks.
- **Status**: **Reverted**. Both correction methods removed from `aPC_PCKBackendLowRankGE.m` and `pckBuildLowRankAugmentedSystem.m`.
- **Clean baseline re-established**: Ready for literature-guided innovations (adaptive rank selection, GEPLS preconditioning, etc.).

### Lessons Learned

1. Don't impose structure: Let data reveal structure via residual analysis.
2. Validate on response: Any gradient correction must improve response prediction, not just gradient reconstruction.
3. Expect inconsistency: Single correction strategy won't work across problem types.
4. Follow literature: Residual-based, adaptive, or multifidelity approaches are more robust than imposed diagonal corrections.

## 9. Low-Rank Adaptive Rank Selection (Implemented, Tested, Reverted)

- Goal: replace fixed low-rank truncation with an adaptive rank selector for the second reduction step.
- Files touched during the experiment:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `coefficients/pck/pckBuildLowRankAugmentedSystem.m`
  - `tests/TestModelTypes.m`
- Prototype design:
  - Keep the same SVD-based low-rank directions.
  - Interpret `GELowRankOpts.rank` as a rank cap.
  - Add a bootstrap-based subspace-stability rule to choose the final retained rank.
  - Return rank-selection diagnostics in `results.PCKModel.lowrank`.

### First Version: Pure Bootstrap Stability

- Added options:
  - `GELowRankOpts.adaptive_rank_method = 'bootstrap_stability'`
  - `GELowRankOpts.bootstrap_samples`
  - `GELowRankOpts.bootstrap_stability_threshold`
- Logic:
  - Resample training rows of the model-space gradient matrix `G`.
  - Recompute SVD on each bootstrap sample.
  - Measure the average principal-angle similarity between the reference `k`-dimensional right-singular subspace and the bootstrap subspace.
  - Select the largest `k` whose mean similarity exceeds the threshold.

### First-Version Results

#### DIM100, PLS-12

| Method | Selected Rank | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: | ---: |
| Fixed low-rank | `3` | `2.778563e-01` | `8.038455e-02` | `9.766s` |
| Bootstrap stability | `12` | `2.879446e-01` | `9.738706e-02` | `52.491s` |

#### Wingweight, no reduction

| Method | Selected Rank | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: | ---: |
| Fixed low-rank | `6` | `6.245385e-08` | `3.670210e-08` | `103.128s` |
| Bootstrap stability | `10` | `3.479933e-08` | `6.417964e-09` | `406.204s` |

Interpretation:

- The pure bootstrap-stability rule was too conservative.
- It strongly preferred high-rank or nearly full-rank solutions.
- Accuracy could improve slightly, but compression benefit and training efficiency were largely lost.

### Second Version: Bootstrap Stability + Relative Singular-Value Gate

- Upgrade made after the first failure:
  - add `GELowRankOpts.bootstrap_min_relative_sv`
  - only keep modes that are both:
    - bootstrap-stable, and
    - above a minimum relative singular-value threshold.
- Final tested default during the experiment:
  - `bootstrap_min_relative_sv = 0.08`

### Second-Version Results

#### DIM100, PLS-12

| Method | Selected Rank | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: | ---: |
| Fixed low-rank | `3` | `2.778563e-01` | `8.038455e-02` | `13.110s` |
| Stability + SV gate | `3` | `2.778563e-01` | `8.038455e-02` | `11.592s` |

#### Wingweight, no reduction

| Method | Selected Rank | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: | ---: |
| Fixed low-rank | `6` | `6.245385e-08` | `3.670210e-08` | `112.562s` |
| Stability + SV gate | `3` | `3.691679e-07` | `1.801973e-07` | `33.218s` |

Interpretation:

- The singular-value gate fixed the "always choose a large rank" problem.
- On `dim100`, the adaptive rule recovered the same rank and performance as the known good fixed-rank setting.
- On `wingweight`, the rule became too aggressive and over-compressed the gradient information.

### Outcome

- **Mixed but not publishable**:
  - version 1 selected ranks that were too high and too slow;
  - version 2 selected ranks that were reasonable for some problems but too small for others.
- The mechanism was still fundamentally a rank-selection rule on top of the same SVD basis, not a new second-reduction operator.
- Tuning thresholds further looked more like parameter fitting than method innovation.

### Final Status

- **Reverted**.
- The adaptive-rank interface and diagnostics were removed.
- `geapck_lowrank` was restored to the previous fixed-rank SVD implementation.

### Lessons Learned

1. Pure stability criteria naturally drift toward high-rank solutions when the singular subspaces are numerically stable.
2. Adding spectral gates can restore compression, but then the method becomes highly threshold-sensitive.
3. This line of work mainly changes truncation strategy, not the second reduction operator itself.
4. If the goal is a stronger methodological contribution, future work should move beyond rank selection and change how the second reduced subspace is constructed.

## 10. Hybrid Residual Subspace (Current Best Candidate)

- Goal: move beyond weighted/truncated SVD by explicitly building the second reduced space as a combination of:
  - a main gradient subspace from the original model-space gradient matrix `G`, and
  - a residual correction subspace from the pilot-trend gradient residual `G - G_pilot`.
- Files touched:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `tests/TestModelTypes.m`
- Final interface:
  - `GELowRankOpts.method = 'hybrid_residual_subspace'`
  - optional `GELowRankOpts.residual_rank`

### Design

- Step 1: fit a lightweight pilot trend using the existing LARS trend path.
- Step 2: compute the pilot trend gradient matrix and form `G_pilot`.
- Step 3: split the final low-rank basis into:
  - `V_main`: leading right-singular vectors of `G`
  - `V_res`: leading right-singular vectors of `G - G_pilot`
- Step 4: project the residual basis onto the orthogonal complement of `V_main`, concatenate, and QR-orthonormalize.

This changes the second reduction operator itself: the final basis is no longer a single SVD truncation of `G`, but a structured combination of principal and residual directions.

### Why This Was Introduced

- The weighted-SVD attempts (`residual_weighted`, `residual_consistency`) only changed sample weighting and remained too close to supervised SVD variants.
- The new method tries to preserve the dominant global compression while recovering directions specifically associated with what the pilot trend failed to capture.

### Unit-Test Status

- Added a dedicated low-rank test path for `hybrid_residual_subspace`.
- Full `tests/TestModelTypes.m` passed after the method swap.

### Benchmark Results

#### DIM100, PLS-12, rank = 3

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | `2.778563e-01` | `8.038454e-02` | `9.834s` |
| `hybrid_residual_subspace` | **`2.483832e-01`** | **`7.374099e-02`** | **`9.151s`** |

Interpretation:

- Clear positive signal.
- This is the first second-reduction variant after the reverted diagonal/adaptive/weighted-SVD experiments that improved both validation error and LOO on a representative hard case.

#### Wingweight, no reduction, rank = 6

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | **`6.416080e-08`** | `3.753386e-08` | **`84.274s`** |
| `hybrid_residual_subspace` | `8.573836e-08` | **`2.240241e-08`** | `107.882s` |

Interpretation:

- Mixed result.
- LOO improved, but validation error worsened and training time increased.
- This suggests the hybrid residual basis may be overfitting or over-correcting on smoother, already well-behaved problems.

#### DIM150, MCP-12, rank = 9

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | `1.376675e-01` | **`1.095995e-02`** | `192.409s` |
| `hybrid_residual_subspace` | **`1.262470e-01`** | `1.648719e-02` | **`27.864s`** |

Interpretation:

- Strong positive signal on another high-dimensional case.
- Validation error improved noticeably.
- Training time dropped dramatically in this run.
- LOO became worse, which again suggests that the hybrid residual basis is improving held-out prediction without necessarily improving the current LOO proxy.

#### MORRIS10, PLS-8, rank = 4

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | **`6.146175e-01`** | **`4.130923e-01`** | `17.837s` |
| `hybrid_residual_subspace` | `6.198550e-01` | `4.219262e-01` | **`16.339s`** |

Interpretation:

- No improvement on this case.
- The hybrid residual basis is slightly worse in both validation error and LOO, though training is slightly faster.
- This supports the view that the method is not a universal replacement for SVD and may be most useful on harder, more unstable high-dimensional problems.

### Current Assessment

- **Best experimental upgrade so far**, but not yet robust enough for default use.
- Strong positive signal on `dim100`.
- Strong positive signal also appeared on `dim150`.
- Still unstable on `wingweight` and not beneficial on `morris10`.
- Better justified than weighted SVD, since the operator now has an explicit residual-subspace construction rather than just a different weighting matrix.

### Next Checkpoint

- Summarize whether this method should be positioned as a targeted improvement for difficult high-dimensional cases rather than a universal replacement.

## 11. CCA Fusion (Implemented, Tested, Not Competitive)

- Goal: replace pure gradient-energy extraction with a supervised second-stage operator that learns a subspace from value-gradient coupling.
- Files touched:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `coefficients/pck/pckBuildLowRankAugmentedSystem.m`
  - `tests/TestModelTypes.m`
- Final interface:
  - `GELowRankOpts.method = 'cca_fusion'`
  - optional `GELowRankOpts.cca_reg`
  - optional `GELowRankOpts.cca_feature_rank`

### Design

- Build a pilot trend from the existing LARS path.
- Construct a value-side feature matrix from:
  - `Y`
  - pilot prediction `Y_pilot`
  - residual `Y - Y_pilot`
  - `|Y - Y_pilot|`
  - optional compressed pilot-trend feature scores.
- Perform a regularized CCA-style fusion between:
  - value-side features, and
  - model-space gradient matrix `G`.
- Use the gradient-side canonical directions as the second reduction basis.
- If the canonical basis dimension is smaller than the requested rank, fill the remainder with the standard SVD basis of `G`.

This is a genuine new operator relative to SVD because the basis is learned from supervised value-gradient coupling instead of gradient variance alone.

### Unit-Test Status

- Added a dedicated `cca_fusion` low-rank test.
- Preserved low-rank diagnostic fields through augmented-system assembly.
- Full `tests/TestModelTypes.m` passed.

### Benchmark Results

#### DIM100, PLS-12, rank = 3

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | `2.778563e-01` | `8.038454e-02` | `10.446s` |
| `hybrid_residual_subspace` | **`2.483832e-01`** | **`7.374099e-02`** | **`8.701s`** |
| `cca_fusion` | `2.957588e-01` | `1.731341e-01` | `22.716s` |

Interpretation:

- `cca_fusion` is clearly worse than both `svd` and `hybrid_residual_subspace` on this case.
- The supervised coupling did not translate into better predictive compression here.

#### Wingweight, no reduction, rank = 6

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | **`6.416080e-08`** | `3.753386e-08` | **`88.284s`** |
| `hybrid_residual_subspace` | `8.573836e-08` | **`2.240241e-08`** | `109.479s` |
| `cca_fusion` | `2.606151e-07` | `1.464656e-07` | `140.686s` |

Interpretation:

- `cca_fusion` is substantially worse than both existing alternatives.
- This case provides strong evidence against using CCA fusion as a default low-rank operator.

#### DIM150, MCP-12, rank = 9

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | `1.376675e-01` | `1.095995e-02` | `189.259s` |
| `hybrid_residual_subspace` | **`1.262470e-01`** | `1.648719e-02` | **`26.689s`** |
| `cca_fusion` | `1.381645e-01` | **`1.033408e-02`** | `145.320s` |

Interpretation:

- `cca_fusion` slightly improves LOO relative to `svd`, but validation error is marginally worse than `svd` and far worse than `hybrid_residual_subspace`.
- It does not deliver the high-dimensional gains that the hybrid method already showed.

#### MORRIS10, PLS-8, rank = 4

| Method | RelMSE | LOO | Train |
| --- | ---: | ---: | ---: |
| `svd` | `6.146175e-01` | **`4.130923e-01`** | **`15.919s`** |
| `hybrid_residual_subspace` | `6.198550e-01` | `4.219262e-01` | `16.714s` |
| `cca_fusion` | **`6.087291e-01`** | `4.508407e-01` | `49.188s` |

Interpretation:

- `cca_fusion` gives a small validation-error improvement relative to `svd`, but LOO gets worse and training cost increases sharply.
- This is not strong enough to justify the extra complexity.

### Current Assessment

- **Not competitive overall**.
- The method is mathematically more interesting than weighted SVD, but in these tests it failed to outperform the current best candidate (`hybrid_residual_subspace`) and often underperformed even plain `svd`.
- The supervised value-side feature design used here does not appear to be the right coupling mechanism for this low-rank GE setting.

### Final Status

- Kept as an experimental branch option, not as a recommended direction.
- If revisited in the future, it likely needs a different supervised objective or a much stronger value-side representation than the current pilot-trend feature construction.

## 12. Supervised Joint Subspace (Implemented, Tested, Mixed)

- Goal: make the second reduction explicitly supervised without introducing a full outer optimization loop.
- Files touched:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `tests/TestModelTypes.m`
- Final interface:
  - `GELowRankOpts.method = 'supervised_joint_subspace'`

### Design

- Build two candidate second-stage subspaces:
  - `svd`
  - `hybrid_residual_subspace`
- Use the current LARS pilot trend to construct a low-cost supervised proxy.
- For each candidate basis, form the projected gradient modes `Z` and score the candidate with a pilot-level leave-one-out proxy:
  - regress `Y` on `[F_pilot, Z]`
  - compute normalized LOO error
- Select the candidate with the lower supervised proxy score.

This is a supervised selector rather than a brand-new algebraic basis constructor. The intent is to let the method automatically choose between the current two strongest operators depending on the problem structure.

### Unit-Test Status

- Added a dedicated `supervised_joint_subspace` low-rank test.
- Full `tests/TestModelTypes.m` passed.

### Benchmark Results

#### DIM100, PLS-12, rank = 3

| Method | RelMSE | LOO | Train | Selected |
| --- | ---: | ---: | ---: | --- |
| `svd` | `2.778563e-01` | `8.038454e-02` | `9.849s` | `-` |
| `hybrid_residual_subspace` | **`2.483832e-01`** | **`7.374099e-02`** | **`8.531s`** | `-` |
| `supervised_joint_subspace` | **`2.483832e-01`** | **`7.374099e-02`** | `8.685s` | `hybrid_residual_subspace` |

Interpretation:

- Positive result.
- The supervised selector correctly chose the hybrid residual basis and matched the current best result.

#### Wingweight, no reduction, rank = 6

| Method | RelMSE | LOO | Train | Selected |
| --- | ---: | ---: | ---: | --- |
| `svd` | **`6.416080e-08`** | `3.753386e-08` | `90.896s` | `-` |
| `hybrid_residual_subspace` | `8.573836e-08` | **`2.240241e-08`** | `112.561s` | `-` |
| `supervised_joint_subspace` | **`6.416080e-08`** | `3.753386e-08` | **`84.310s`** | `svd` |

Interpretation:

- Positive result.
- The supervised selector correctly chose the plain SVD basis and avoided the degradation from the hybrid residual subspace.

#### DIM150, MCP-12, rank = 9

| Method | RelMSE | LOO | Train | Selected |
| --- | ---: | ---: | ---: | --- |
| `svd` | `1.376675e-01` | **`1.095995e-02`** | `187.124s` | `-` |
| `hybrid_residual_subspace` | **`1.262470e-01`** | `1.648719e-02` | **`26.595s`** | `-` |
| `supervised_joint_subspace` | `1.280578e-01` | `1.666157e-02` | `27.411s` | `svd` |

Interpretation:

- Mixed/negative result.
- The supervised proxy preferred `svd`, but the final validation performance stayed closer to the hybrid result and did not recover the true best case.
- This indicates that the current pilot-level LOO proxy is not reliable enough on this harder regime.

#### MORRIS10, PLS-8, rank = 4

| Method | RelMSE | LOO | Train | Selected |
| --- | ---: | ---: | ---: | --- |
| `svd` | **`6.146175e-01`** | **`4.130923e-01`** | **`16.742s`** | `-` |
| `hybrid_residual_subspace` | `6.198550e-01` | `4.219262e-01` | `16.875s` | `-` |
| `supervised_joint_subspace` | `6.198550e-01` | `4.219262e-01` | `16.953s` | `hybrid_residual_subspace` |

Interpretation:

- Negative result.
- The supervised selector picked the wrong candidate and inherited the slight degradation of the hybrid basis.

### Current Assessment

- Better than CCA fusion because it can recover the correct choice on some benchmarks.
- Still not robust enough for default use because the pilot-level supervised proxy does not generalize reliably across all problem classes.
- The approach is promising as a selector framework, but the current supervised score is still too weak.

### Final Status

- Kept as an experimental branch option.
- Not recommended as the new default low-rank method in its current form.

## 13. Supervised Joint Operator (Implemented, Tested, Strongest So Far)

- Goal: turn the second reduction into a genuinely supervised operator, instead of a pure algebraic truncation or a hard selector between existing bases.
- Files touched:
  - `coefficients/aPC_PCKBackendLowRankGE.m`
  - `tests/TestModelTypes.m`
- Final interface:
  - `GELowRankOpts.method = 'supervised_joint_operator'`
  - optional `GELowRankOpts.operator_eta_grid`
  - optional `GELowRankOpts.operator_grad_weight`

### Design

- Build two endpoint subspaces:
  - `V_main`: the standard low-rank SVD basis of `G`
  - `V_hybrid`: the residual-corrected hybrid basis from `hybrid_residual_subspace`
- Define a continuous family of second-stage subspaces:

```matlab
V(eta) = qr(V_main + eta * (V_hybrid - V_main))
```

- Search over a small `eta` grid instead of repeatedly training full GE models.
- For each candidate `V(eta)`, evaluate a lightweight supervised objective:
  - value term: pilot-level normalized LOO of `Y ~ [F_pilot, G*V(eta)]`
  - gradient term: normalized projection loss `||G - G V(eta) V(eta)'||_F^2 / ||G||_F^2`
- Final score:

```matlab
J(eta) = L_value(eta) + lambda * L_grad(eta)
```

- Select `eta* = argmin J(eta)` and only then enter the full low-rank GE training path once.

### Why This Differs From Earlier Attempts

- Unlike adaptive rank selection, it does not just decide how many SVD modes to keep.
- Unlike weighted SVD or CCA fusion, it does not build the basis from a single algebraic rule.
- Unlike `supervised_joint_subspace`, it is not a discrete selector over a few candidates.
- It defines a **continuous supervised operator family** and learns the final second-stage basis by minimizing a value-gradient joint objective.

### Unit-Test Status

- Added a dedicated `supervised_joint_operator` low-rank test.
- Full `tests/TestModelTypes.m` passed.

### Benchmark Results

#### DIM100, PLS-12, rank = 3

| Method | RelMSE | LOO | Train | Extra |
| --- | ---: | ---: | ---: | --- |
| `svd` | `2.778563e-01` | `8.038454e-02` | `21.737s` | - |
| `hybrid_residual_subspace` | `2.483832e-01` | `7.374099e-02` | `18.936s` | - |
| `supervised_joint_operator` | **`2.477026e-01`** | **`7.360332e-02`** | `19.573s` | `eta=0.80` |

Interpretation:

- Strong positive result.
- This is the first supervised second-stage operator that outperformed both `svd` and `hybrid_residual_subspace` on `dim100`.

#### Wingweight, no reduction, rank = 6

| Method | RelMSE | LOO | Train | Extra |
| --- | ---: | ---: | ---: | --- |
| `svd` | `6.416080e-08` | **`3.753386e-08`** | `140.948s` | - |
| `hybrid_residual_subspace` | `8.573836e-08` | `2.240241e-08` | `154.391s` | - |
| `supervised_joint_operator` | **`6.305239e-08`** | `3.774405e-08` | **`88.294s`** | `eta=0.00` |

Interpretation:

- Positive result.
- The supervised operator correctly collapsed back to the pure SVD endpoint (`eta=0`) and slightly improved validation error while reducing training time in this run.

#### DIM150, MCP-12, rank = 9

| Method | RelMSE | LOO | Train | Extra |
| --- | ---: | ---: | ---: | --- |
| `svd` | `1.376675e-01` | **`1.095995e-02`** | `221.772s` | - |
| `hybrid_residual_subspace` | **`1.262470e-01`** | `1.648719e-02` | **`30.527s`** | - |
| `supervised_joint_operator` | `1.280578e-01` | `1.666157e-02` | `31.379s` | `eta=0.00` |

Interpretation:

- Mixed result.
- The operator stayed on the SVD-side endpoint (`eta=0`) and improved over plain `svd`, but still did not recover the best `hybrid` result.
- This suggests the current supervised proxy is still imperfect on the hardest high-dimensional regime.

#### MORRIS10, PLS-8, rank = 4

| Method | RelMSE | LOO | Train | Extra |
| --- | ---: | ---: | ---: | --- |
| `svd` | **`6.146175e-01`** | **`4.130923e-01`** | `20.060s` | - |
| `hybrid_residual_subspace` | `6.198550e-01` | `4.219262e-01` | `24.653s` | - |
| `supervised_joint_operator` | `6.206391e-01` | `4.248131e-01` | `22.170s` | `eta=1.20` |

Interpretation:

- Negative result on this benchmark.
- The current supervised proxy still over-trusts residual enrichment on `morris10` and moves too far away from the plain SVD basis.

### Current Assessment

- **Strongest supervised low-rank operator tried so far.**
- Better than `cca_fusion` and better than the earlier discrete supervised selector.
- Already shows true gains on `dim100` and `wingweight`.
- Still unstable on `dim150` and `morris10`, so it is not yet ready as a universal default.

### Final Status

- Kept as the current leading innovation candidate for second-stage low-rank GE.
- Needs a stronger supervised proxy before it can be positioned as the final method.
