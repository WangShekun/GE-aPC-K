# Gradient-Aware Reduction Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gradient-enhanced PLS and active-subspace-plus-PLS reduction methods that can participate in GEAPCK training.

**Architecture:** Precompute response gradients in standardized U-space before reduction whenever the model or reduction method needs them. Add two new reduction methods under `dimreduct/`, both returning explicit linear maps so `aPC_PrepareModelGradients` can propagate gradients through them with the existing explicit method dispatch pattern.

**Tech Stack:** MATLAB, `plsregress`, existing GEAPCK pipeline, MATLAB unit tests (`runtests`)

---

## File Map

- Create: `dimreduct/aPC_dimReduction_ge_pls.m`
- Create: `dimreduct/aPC_dimReduction_as_pls.m`
- Create: `utils/aPC_PrepareUGradients.m`
- Modify: `apc_train/aPC_TrainCandidate.m`
- Modify: `dimreduct/aPC_dimReduction.m`
- Modify: `utils/aPC_PrepareModelGradients.m`
- Modify: `tests/TestModelTypes.m`
- Modify: `README.md`

### Task 1: Add failing tests

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Add tests that `ge_pls` and `as_pls` error when gradients are unavailable.
- [ ] Add tests that `ge_pls` and `as_pls` work with provided gradients in GEAPCK and produce reduced gradients of the expected size.
- [ ] Run the focused tests and confirm they fail before implementation.

### Task 2: Precompute U-space gradients

**Files:**
- Create: `utils/aPC_PrepareUGradients.m`
- Modify: `apc_train/aPC_TrainCandidate.m`
- Modify: `utils/aPC_PrepareModelGradients.m`

- [ ] Extract the shared `dY/dX -> dY/dU` preparation logic into a reusable helper.
- [ ] Call it before reduction when either GEAPCK or a gradient-aware reduction method requires gradients.
- [ ] Make `aPC_PrepareModelGradients` reuse precomputed `dYdU_train` when present.

### Task 3: Implement new reduction methods

**Files:**
- Create: `dimreduct/aPC_dimReduction_ge_pls.m`
- Create: `dimreduct/aPC_dimReduction_as_pls.m`
- Modify: `dimreduct/aPC_dimReduction.m`
- Modify: `utils/aPC_PrepareModelGradients.m`

- [ ] Add `ge_pls` using multivariate PLS on `[Y, weighted standardized dYdU]`.
- [ ] Add `as_pls` using an active-subspace basis from gradients followed by PLS in that subspace.
- [ ] Return explicit linear maps (`W` / total map) so GE gradient propagation remains linear.
- [ ] Extend explicit reduction-method dispatch in `aPC_PrepareModelGradients` for the new methods.

### Task 4: Verify and document

**Files:**
- Modify: `README.md`
- Verify: `tests/TestModelTypes.m`

- [ ] Run focused new tests.
- [ ] Run `runtests('tests/TestModelTypes.m')`.
- [ ] Add a short README note about the new reduction methods and their gradient requirement.
