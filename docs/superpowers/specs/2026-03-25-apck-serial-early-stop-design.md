## Goal

Add an optional early-stopping rule to the serial `optimal` APCK/GEAPCK trend search: stop after 3 consecutive trend expansions that do not improve the best-so-far LOO by more than a tolerance.

## Scope

- Apply only to `PCKType='optimal'` with serial trend evaluation.
- Do not change trend-parallel behavior.
- Compare each candidate against the historical best LOO, not just the immediately previous candidate.
- Use a tolerance to ignore harmless LOO noise.

## Design

- Add two optional `KrigingOpts` fields in `coefficients/aPC_PCKBackend.m`:
  - `serial_early_stop_patience` (default `Inf`, disabled)
  - `loo_improve_tol` (default `1e-4`)
- In the serial `optimal` loop, track:
  - `best.loo`
  - `no_improve_count`
- Treat a candidate as an improvement only if:

```matlab
model_i.loo < best.loo - loo_improve_tol
```

- Reset the counter on improvement; otherwise increment it.
- Break the serial loop once `no_improve_count >= serial_early_stop_patience`.
- Leave `sequential` and trend-parallel paths unchanged.

## Verification

- Add a focused unit test showing serial optimal search stops early when tolerance is set large enough to suppress later improvements.
- Keep existing trend-parallel tests unchanged to confirm parallel behavior still evaluates the full candidate set.
