# GEAPCK Augmented System Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GEAPCK augmented-system assembly overhead by replacing large matrix concatenations with direct preallocated writes.

**Architecture:** Keep the mathematical system unchanged and only change how `pckBuildAugmentedSystem` assembles `K`. Add a focused equivalence test around the augmented system output, then replace `cell2mat` and `[Kff, Kfg; Kgf, Kgg]` construction with preallocated block writes into the final dense matrix.

**Tech Stack:** MATLAB, MATLAB unit tests (`runtests`)

---

## File Map

- Modify: `coefficients/pck/pckBuildAugmentedSystem.m`
- Modify: `tests/TestModelTypes.m`

### Task 1: Add a failing equivalence test

**Files:**
- Modify: `tests/TestModelTypes.m`

- [ ] Add a focused test that assembles a small GE augmented system from known `U`, `Y`, `G`, `F`, `dF`, `theta` inputs and checks exact expected block layout.
- [ ] Run the focused test and confirm it fails before the implementation change.

### Task 2: Replace concatenation with preallocated writes

**Files:**
- Modify: `coefficients/pck/pckBuildAugmentedSystem.m`

- [ ] Preallocate the final `K` matrix with size `N * (d + 1)`.
- [ ] Fill the `Kff`, `Kfg`, `Kgf`, and `Kgg` blocks directly into `K` without `cell2mat` or whole-matrix concatenation.
- [ ] Keep `y_aug` and `H` outputs unchanged.

### Task 3: Verify behavior and check performance directionally

**Files:**
- Verify: `tests/TestModelTypes.m`
- Verify: profiler or benchmark script as needed

- [ ] Run the focused equivalence test.
- [ ] Run `runtests('tests/TestModelTypes.m')`.
- [ ] Re-run the relevant GEAPCK profiling case or timing command and compare whether `pckBuildAugmentedSystem` time moves down directionally.
