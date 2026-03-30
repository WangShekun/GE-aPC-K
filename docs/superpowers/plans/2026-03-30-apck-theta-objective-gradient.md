# APCK Theta Objective Gradient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analytic objective gradient for APCK `theta` optimization and measure whether it improves `fmincon` behavior.

**Architecture:** Keep GEAPCK unchanged and only add the gradient-enabled path for non-GE APCK. Extend the APCK negative log-likelihood evaluation so it can return `dNLL/dlog(theta)`, then enable `SpecifyObjectiveGradient` in `fmincon` for that branch and compare optimizer diagnostics on a small benchmark.

**Tech Stack:** MATLAB, `fmincon`, MATLAB unit tests (`runtests`)

---

## File Map

- Modify: `coefficients/aPC_PCKBackend.m`
- Modify: `tests/TestModelTypes.m`
- Verify: MATLAB one-off benchmark command

### Task 1: Add failing tests for the APCK gradient path

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Add a focused APCK test that runs `optimize_theta=true` and verifies the optimizer reports the analytic-gradient path.
- [ ] Add a focused APCK test that checks the gradient-enabled objective still produces finite diagnostics and bounded `theta`.
- [ ] Run the focused tests and confirm they fail before implementation.

### Task 2: Implement APCK-only objective gradients

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`

- [ ] Extend the APCK NLL evaluation to optionally return the objective gradient with respect to `log(theta)`.
- [ ] Keep GEAPCK on the existing objective-only path.
- [ ] Enable `SpecifyObjectiveGradient=true` only for the APCK branch that returns the gradient.
- [ ] Preserve current fallback behavior when Cholesky fails.

### Task 3: Verify correctness and measure effect

**Files:**
- Verify: `tests/TestModelTypes.m`
- Verify: one-off MATLAB benchmark command

- [ ] Run `runtests('tests/TestModelTypes.m')`.
- [ ] Run a small APCK benchmark twice if needed to inspect optimizer diagnostics such as function evaluations / iterations.
- [ ] Summarize whether the analytic gradient reduced optimizer work in the tested case.
