# APCK Serial Early Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional early stopping to the serial `optimal` APCK/GEAPCK trend search without changing parallel behavior.

**Architecture:** Extend `coefficients/aPC_PCKBackend.m` with two optional `KrigingOpts` defaults and a best-so-far early-stop counter inside the serial `optimal` loop. Guard the feature behind configuration so existing behavior stays unchanged unless the new option is set.

**Tech Stack:** MATLAB, MATLAB unit tests (`runtests`)

---

## File Map

- Modify: `coefficients/aPC_PCKBackend.m`
- Modify: `tests/TestModelTypes.m`
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-03-25-apck-serial-early-stop-design.md`

### Task 1: Add a failing regression test

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Add a serial `optimal` APCK test with `DisplayProgress=true`, `max_trend_terms=[1, 4]`, `serial_early_stop_patience=2`, and a very large `loo_improve_tol`.
- [ ] Run the single test and confirm it fails because serial search still evaluates `trend 4/4` before the implementation change.

### Task 2: Implement serial-only early stopping

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`

- [ ] Add defaults for `KrigingOpts.serial_early_stop_patience` and `KrigingOpts.loo_improve_tol`.
- [ ] In the serial `optimal` loop, track a no-improvement counter against `best.loo`.
- [ ] Count an improvement only when `model_i.loo < best.loo - loo_improve_tol`.
- [ ] Break once the consecutive no-improvement count reaches the configured patience.
- [ ] Leave the `sequential` branch and trend-parallel branch unchanged.

### Task 3: Verify and document

**Files:**
- Modify: `README.md`
- Verify: `tests/TestModelTypes.m`

- [ ] Run the focused test file and confirm it passes.
- [ ] Update README text describing `sequential`, `optimal`, and the new serial early-stop option.
- [ ] Run GitNexus change detection to confirm the scope remains limited.
