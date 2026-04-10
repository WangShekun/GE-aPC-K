# Remove use_objective_gradient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `KrigingOpts.use_objective_gradient` from APCK/GEAPCK theta optimization while keeping theta optimization behavior working without objective gradients.

**Architecture:** The change is localized to the PCK theta optimizer and its diagnostics. Remove the flag-driven optimizer branch, keep the non-objective-gradient `fmincon` path, and then clean tests, scripts, and README so no public surface mentions the deleted feature.

**Tech Stack:** MATLAB, GitNexus, MATLAB unit tests (`runtests`)

---

### Task 1: Lock in the new behavior with tests

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Replace one objective-gradient-focused test with a failing test that asserts theta optimization no longer exposes `theta_uses_objective_gradient` diagnostics.
- [ ] Run the targeted MATLAB test and confirm it fails for the expected reason.

### Task 2: Remove backend support

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`

- [ ] Remove `KrigingOpts.use_objective_gradient` defaults and all optimizer branching that depends on it.
- [ ] Keep theta optimization on the existing non-objective-gradient `fmincon` path.
- [ ] Remove result fields that only existed for objective-gradient diagnostics.

### Task 3: Remove public-facing remnants

**Files:**
- Modify: `tests/TestModelTypes.m`
- Modify: `main_compare.m`
- Modify: legacy GEAPCK compare script (removed)
- Modify: `README.md`

- [ ] Delete or rewrite tests that mention the removed option.
- [ ] Remove the option from example scripts.
- [ ] Remove documentation references so the public API no longer advertises the flag.

### Task 4: Verify the reduced surface still works

**Files:**
- Test: `tests/TestModelTypes.m`

- [ ] Run targeted MATLAB tests covering APCK/GEAPCK theta optimization.
- [ ] Confirm no remaining repository references to `use_objective_gradient`.
- [ ] Summarize the final blast radius and verification evidence.
