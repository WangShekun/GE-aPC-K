# APCK Sequential Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy sequential-only dead state from `aPC_PCKBackend` while preserving the current APCK sequential semantics and regression coverage.

**Architecture:** Keep the current backend structure and make a minimal cleanup in `coefficients/aPC_PCKBackend.m`. The sequential branch continues to fit exactly once on `final_support`, while the optimal branch continues to scan trend sizes. Verification stays centered on `tests/TestModelTypes.m`.

**Tech Stack:** MATLAB, MATLAB unit tests (`runtests`), Git

---

## File Map

- Modify: `coefficients/aPC_PCKBackend.m` — remove dead state and branches tied to the old sequential meaning.
- Verify: `tests/TestModelTypes.m` — keep the existing regression that asserts sequential uses the final LARS support.
- Reference: `docs/superpowers/specs/2026-03-24-apck-sequential-cleanup-design.md` — approved design scope.

### Task 1: Preserve sequential regression coverage

**Files:**
- Test: `tests/TestModelTypes.m`

- [ ] **Step 1: Inspect the existing sequential APCK regression**

Confirm that `tests/TestModelTypes.m` already asserts `sort(results.PCKModel.trend_idx) == sort(results.lars_idx(:)')` for the sequential APCK path.

- [ ] **Step 2: Decide whether a new failing test is necessary**

If the assertion already exists and directly captures the intended invariant, keep the test file unchanged.

- [ ] **Step 3: Run the focused regression before code changes**

Run: `matlab -batch "results = runtests('tests/TestModelTypes.m'); assertSuccess(results);"`

Expected: the file runs successfully before cleanup, establishing a baseline.

### Task 2: Remove obsolete sequential carry-forward state

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`

- [ ] **Step 1: Delete dead state initialization**

Remove `theta_prev = [];` from the top-level candidate-selection block.

- [ ] **Step 2: Delete dead sequential carry-forward branch**

Inside the serial trend loop, remove:

```matlab
            theta_init = [];
            if strcmp(pck_type, 'sequential') && ~isempty(theta_prev)
                theta_init = theta_prev;
            end
```

Replace it with the minimal direct initialization needed by the optimal path:

```matlab
            theta_init = [];
```

- [ ] **Step 3: Delete dead theta write-back**

Remove:

```matlab
            theta_prev = model_i.theta;
```

because the sequential branch no longer iterates through nested trends.

- [ ] **Step 4: Re-read the branch structure for semantic safety**

Confirm the resulting top-level behavior is still:

- `sequential` -> one call to `localEvaluateTrendIdx(final_support, ...)`
- `optimal` + trend parallel -> parallel scan of `trend_sizes`
- otherwise -> serial scan of `trend_sizes`

### Task 3: Verify behavior did not change

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`
- Test: `tests/TestModelTypes.m`

- [ ] **Step 1: Run the focused model-type test file**

Run: `matlab -batch "results = runtests('tests/TestModelTypes.m'); assertSuccess(results);"`

Expected: PASS.

- [ ] **Step 2: Confirm the dead symbol is gone**

Run: `rg "theta_prev" coefficients/aPC_PCKBackend.m`

Expected: no matches.

- [ ] **Step 3: Review git diff for scope**

Run: `git diff -- coefficients/aPC_PCKBackend.m tests/TestModelTypes.m`

Expected: only the intended cleanup in `coefficients/aPC_PCKBackend.m`, with tests either unchanged or minimally adjusted.

- [ ] **Step 4: Run GitNexus change detection before any commit**

Run the MCP tool: `gitnexus_detect_changes({scope: "all", repo: "GE-aPC-K"})`

Expected: changed scope remains low risk and limited to the intended files.

### Task 4: Commit cleanly

**Files:**
- Modify: `coefficients/aPC_PCKBackend.m`
- Optionally modify: `tests/TestModelTypes.m`

- [ ] **Step 1: Stage only relevant files**

Run: `git add coefficients/aPC_PCKBackend.m tests/TestModelTypes.m`

- [ ] **Step 2: Commit with a minimal message**

Run:

```bash
git commit -m "refactor: remove obsolete APCK sequential theta state"
```

- [ ] **Step 3: Re-run status**

Run: `git status --short`

Expected: no unintended staged leftovers from this cleanup.
