# AGENTS

Guide for agentic coding in this MATLAB aPC repository.

## Scope
- MATLAB codebase for Arbitrary Polynomial Chaos Expansion (aPC).
- Root: `/home/kun/scholar/PCE/aPC/matlab/aPC`.

## Quick Start
- Open MATLAB with working directory set to this repo.
- Initialize paths by running `aPC` (from `core/aPC.m`).
- Run the example script `main.m`.

## Build / Lint / Test
### Build
- No build step; MATLAB scripts and functions run directly.

### Lint
- No configured linter in this repository.
- If you use MATLAB Code Analyzer, keep existing warnings unless you touch the code.

### Tests
- No automated unit test suite is present.
- Use the example script to validate changes:
  - In MATLAB: `run("main.m")`.
- For a single focused check, execute only the function you changed from the MATLAB prompt or a small scratch script.
- If you add tests, prefer MATLAB `unittest` and document the command here.

## Running a Single Example / Script
- Example (Ishigami workflow):
  - `run("main.m")` uses `ishigami_Input` and `ishigami`.
- If you need to run just a benchmark function, call it with inputs:
  - `y = ishigami(x)` where `x` is N x 3.

## Dependencies
- MATLAB (R2020b+ recommended, but not enforced).
- UQLab is required for parts of `main.m` (`uqlab`, `uq_createInput`, `uq_createModel`).
- Parallel toolbox is used when `ModelOpts.Parallel = true`.

## Repository Layout
- `core/` path setup via `aPC.m`.
- `apc_train/` training pipeline (`aPC_Train`).
- `basis/` polynomial basis construction.
- `coefficients/` regression solvers (OLS/LARS/Ridge/Lasso).
- `dimreduct/` dimension reduction (PCA/PLS/KPCA/SKPCA/MCP/ICA/RP).
- `predict/` prediction pipeline.
- `UQ/` uncertainty quantification.
- `sensitivity/` Sobol indices.
- `benchmark/` benchmark functions and input options.
- `display/` plotting/visualization.
- `utils/` utility helpers.

## Code Style Guidelines
### File and Function Naming
- MATLAB function name matches filename.
- Core API functions are prefixed `aPC_` and use PascalCase after the prefix.
- Benchmark functions use lower-case names (e.g., `ishigami`).
- Keep new filenames consistent with these patterns.

### Imports / Paths
- Do not hardcode absolute paths.
- Use `aPC` to manage `addpath/genpath` instead of ad-hoc path edits.
- If a new module is added, update `core/aPC.m` module list.

### Formatting
- Use 4 spaces for indentation.
- Keep line width reasonable (~100 chars), wrap long `fprintf`/strings.
- Use `%%` section headers for major blocks, matching existing files.
- Keep comments concise; do not add comments for obvious MATLAB operations.

### Types and Shapes
- Use numeric arrays; most APIs expect double precision.
- Respect data shapes:
  - Inputs `X_*` are N x M.
  - Outputs `Y_*` are N x 1.
- Preserve struct field shapes, especially `ModelOpts`, `data`, and `results`.

### Struct Fields and Options
- Use `ModelOpts` fields with existing names and casing:
  - `Order`, `qnorm`, `RegressionMethod`, `ReductionMethod`, `red_dim`, `Parallel`.
- Use `isfield` + `isempty` checks before accessing optional fields.
- If you introduce new options, ensure `aPC_build_search_grid` supports them.

### Regression and Methods
- `RegressionMethod` is case-insensitive; keep the stored value consistent.
- When printing, use `upper(...)` to match existing logs.
- Keep LOO as the primary model selection criterion.

### Error Handling
- Prefer explicit validation with `isfield`, `isempty`, and size checks.
- Use `error(...)` for invalid configuration.
- Use `warning(...)` for missing optional modules or non-fatal issues.
- Do not swallow exceptions unless you print a clear message.

### Parallelism
- Respect `ModelOpts.Parallel`.
- When using `parfor`, keep outputs deterministic and collect results in preallocated structs.
- Use `parallel.pool.DataQueue` for progress updates as in `aPC_Train`.

### Determinism
- When example scripts need randomness, set the RNG with `rng(...)`.
- Avoid changing the global RNG in library functions.

### Plotting
- Plotting functions live under `display/`.
- Do not hardcode figure numbers; let MATLAB manage figures.
- Keep plotting optional and callable from the main pipeline.

## Documentation
- Update `README.md` if you add or change user-facing features.
- Keep `CHANGELOG.md` entries short and actionable.

## Common Workflows
- Training: `aPC_Train` -> `aPC_Predict` -> `aPC_ComputeUQ` / `aPC_ComputeSensitivity`.
- Input creation: use `benchmark/inputopts/*_Input.m` with `uq_createInput`.
 - Grid search + model selection: `aPC_Train` builds a search grid from `ModelOpts` (e.g., `Order`, `qnorm`, `red_dim`, kernel params) and selects the configuration with the smallest LOO error.

## Notes for Agents
- There are no Cursor or Copilot instruction files in this repository.
- If you add a new external dependency, document it here and in `README.md`.
- Prefer minimal diffs; do not reformat unrelated code.

## Suggested Commands (copy/paste)
- Initialize paths:
  - `aPC`
- Run the example:
  - `run("main.m")`
- Single function call (Ishigami):
  - `x = rand(10,3); y = ishigami(x);`

## If You Add Tests
- Use MATLAB `unittest` and place tests under `tests/`.
- Document commands:
  - All tests: `results = runtests("tests");`
  - Single test file: `results = runtests("tests/TestName.m");`

## Safe Defaults for Changes
- Keep APIs backward-compatible; update `aPC_Train` and `aPC_Predict` together.
- Any new `ModelOpts` field should have a default in `aPC_Train`.
- Preserve existing printed output formats where possible.

## Quality Checklist
- Paths initialize with `aPC`.
- `main.m` runs without manual path hacks.
- Updated functions handle missing optional fields.
- Outputs keep expected shapes and field names.
- No new warnings/errors from MATLAB unless unavoidable.

End of AGENTS.md.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **GE-aPC-K** (112 symbols, 96 relationships, 0 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/GE-aPC-K/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/GE-aPC-K/context` | Codebase overview, check index freshness |
| `gitnexus://repo/GE-aPC-K/clusters` | All functional areas |
| `gitnexus://repo/GE-aPC-K/processes` | All execution flows |
| `gitnexus://repo/GE-aPC-K/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
