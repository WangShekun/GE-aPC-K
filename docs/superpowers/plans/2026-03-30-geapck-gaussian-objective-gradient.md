# GEAPCK Gaussian Objective Gradient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analytic `fmincon` objective gradient for GEAPCK when the correlation kernel is Gaussian.

**Architecture:** Extend the PCK backend so the GE theta optimizer can provide `dNLL/dlog(theta)` for the Gaussian augmented covariance system only. Leave Matern32 and other GE kernels on the existing objective-only path, and verify the Gaussian path through focused tests and optimizer metadata.

**Tech Stack:** MATLAB, `fmincon`, MATLAB unit tests (`runtests`)

---

## File Map

- Modify: `coefficients/aPC_PCKBackend.m`
- Modify: `tests/TestModelTypes.m`

### Task 1: Add failing GEAPCK tests

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Add a GEAPCK Gaussian test that expects the objective-gradient optimizer path to be enabled.
- [ ] Add a GEAPCK Matern32 test that expects the optimizer to stay on the non-gradient path.
- [ ] Run the focused tests and confirm the Gaussian test fails before implementation.

### Task 2: Implement Gaussian-only GE objective gradients

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`

- [ ] Enable `SpecifyObjectiveGradient` for GE only when `corr='gaussian'`.
- [ ] Implement augmented-system `dK/dtheta` for Gaussian GE blocks.
- [ ] Reuse the existing NLL core outputs to compute `dNLL/dlog(theta)`.
- [ ] Leave non-Gaussian GE kernels on the existing objective-only path.

### Task 3: Verify behavior

**Files:**
- Verify: `tests/TestModelTypes.m`

- [ ] Run the focused GE tests.
- [ ] Run `runtests('tests/TestModelTypes.m')`.
- [ ] Summarize whether the Gaussian GE path now reports objective-gradient usage while Matern32 does not.
