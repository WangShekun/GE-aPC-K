## Goal

Clean up `coefficients/aPC_PCKBackend.m` after the APCK `sequential` path was redefined to mean a single fit on `final_support`, and add regression coverage for that meaning.

## Current Context

- `sequential` no longer iterates through nested trend sizes.
- It now directly evaluates `final_support` through `localEvaluateTrendIdx(...)`.
- The backend still contains legacy state tied to the older behavior, notably `theta_prev` and its carry-forward branch inside the serial trend loop.
- The repository already has a regression test asserting that the selected sequential trend matches `results.lars_idx`.

## Scope

Included:

- Remove dead or redundant state in `coefficients/aPC_PCKBackend.m` that only served the old sequential semantics.
- Keep the current external meaning of `PCKType = 'sequential'` unchanged.
- Keep the existing `optimal` and trend-parallel behavior unchanged.
- Keep or adjust tests in `tests/TestModelTypes.m` so the new sequential meaning remains covered.

Excluded:

- No broader refactor of PCK backend structure.
- No changes to output field schema unless required by the cleanup.
- No unrelated edits in `main_compare.m` or other files.

## Design

### Backend cleanup

In `coefficients/aPC_PCKBackend.m`:

- Delete `theta_prev` initialization.
- Delete the serial-loop branch that conditionally forwards `theta_prev` into `theta_init`.
- Delete the assignment that writes the current model theta back into `theta_prev`.

This leaves two clean execution modes:

- `sequential`: one direct fit on `final_support`
- `optimal`: evaluate candidate trend sizes and select the best one

### Test coverage

Preserve the current regression intent in `tests/TestModelTypes.m`:

- `sequential` should continue to select the full final LARS support rather than an intermediate nested trend.

If the current test already captures that behavior adequately, no extra test expansion is required.

## Risk Assessment

- Expected risk: low.
- Primary caller is `apc_train/aPC_TrainCandidate.m`.
- No execution flows were detected by GitNexus for this backend.
- Behavior risk is limited because the removed state is no longer read on the `sequential` path.

## Verification

- Run the focused MATLAB test file: `runtests("tests/TestModelTypes.m")`.
- Confirm `coefficients/aPC_PCKBackend.m` no longer contains `theta_prev`.
- Confirm `sequential` behavior remains the same through existing APCK test coverage.
