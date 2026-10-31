# AGENTS

Execution guide for agentic coding in this MATLAB aPC repository.

## Scope
- Repository: MATLAB framework for Arbitrary Polynomial Chaos Expansion (aPC).
- Workspace root: `/home/kun/scholar/PCE/aPC/matlab/GE-aPC-K`.
- Focus areas: APC, APCK, GEAPCK, dimension reduction, prediction, UQ, sensitivity.

## Environment and Dependencies
- MATLAB is required. MATLAB R2020b+ is recommended.
- UQLab is required for example scripts and benchmark input construction.
- Parallel Computing Toolbox is required when `ModelOpts.Parallel = 1`.
- Optimization Toolbox is used by the APCK/GEAPCK `theta` optimizer (`fmincon`).

## Quick Start
- Open MATLAB with the repository as the working directory.
- Initialize repository paths by running `aPC`.
- If using benchmark inputs or UQLab baseline models, run `uqlab`.
- Main single-run script: `run("main.m")`.
- Main comparison script: `run("main_compare.m")`.

## Build, Lint, and Test
### Build
- No build step exists. MATLAB functions and scripts run directly.

### Lint
- No standalone linter is configured.
- Use MATLAB Code Analyzer when editing touched files.
- Do not clean unrelated warnings in untouched files.

### Test Commands
- Run the current focused unit test file:
  - `results = runtests("tests/TestModelTypes.m");`
- Run all tests under `tests/`:
  - `results = runtests("tests");`
- Run one named test from a file:
  - `results = runtests("tests/TestModelTypes.m", "ProcedureName", "testAPCKThetaOptimizationUsesFmincon");`
- Run the example workflow manually:
  - `run("main.m")`
- Run the benchmark comparison script:
  - `run("main_compare.m")`

### Validation Expectations
- Prefer `runtests("tests/TestModelTypes.m")` for targeted verification after APC/APCK/GEAPCK changes.
- If you touch user-facing scripts, also run the relevant script (`main.m` or `main_compare.m`) when feasible.
- Before claiming completion, verify the exact file or workflow you changed.

## Repository Layout
- `core/` path initialization via `aPC.m`.
- `apc_train/` model training pipeline and search orchestration.
- `basis/` polynomial basis generation and `Psi` construction.
- `coefficients/` regression backends and PCK/GEAPCK fitting logic.
- `coefficients/pck/` shared PCK kernel helpers.
- `dimreduct/` dimension reduction methods.
- `predict/` prediction pipeline.
- `UQ/` uncertainty quantification post-processing.
- `sensitivity/` Sobol sensitivity analysis.
- `benchmark/` benchmark functions, inputs, and gradients.
- `display/` plotting helpers.
- `utils/` printing, gradients, and general helpers.

## Naming and File Conventions
- MATLAB function name must match filename.
- Core public API functions use `aPC_` prefix and PascalCase after the prefix.
- Benchmark functions use lower-case names, e.g. `ishigami`, `wingweight`, `morris10`.
- Keep new filenames consistent with nearby modules.
- Split files only when it improves clarity; avoid over-fragmenting tightly coupled logic.

## Formatting Rules
- Use 4 spaces for indentation.
- Keep lines reasonably short; wrap long `fprintf` and long expressions.
- Use `%%` section headers in scripts and large MATLAB files where it matches existing style.
- Prefer concise comments that explain non-obvious intent, not obvious MATLAB syntax.
- Preserve existing output formatting unless deliberately improving a reporting surface.

## Data and Type Conventions
- Use numeric arrays, typically `double`.
- Input matrices are `N x M`.
- Output vectors are `N x 1`.
- Gradients for GEAPCK are `N x M`, where column `j` is `dY/dX_j`.
- Preserve struct-based APIs: `ModelOpts`, `data`, `results`, `ReduceModel`.
- Do not silently reshape user data into incompatible conventions.

## ModelOpts Conventions
- Core fields commonly used:
  - `Order`
  - `qnorm`
  - `aPC_Method`
  - `RegressionMethod`
  - `ModelType`
  - `Parallel`
- PCK fields:
  - `PCKType`
  - `KrigingOpts.*`
- GE fields:
  - `GradientMode`
  - `FDOpts.*`
- Reduction fields:
  - `ReductionMethod`
  - `red_dim`
- New options must have safe defaults and must be compatible with `aPC_build_search_grid` if searchable.

## APC / APCK / GEAPCK Rules
- `ModelType = 'apc'` uses polynomial regression only.
- `ModelType = 'apck'` uses polynomial trend plus Kriging residual.
- `ModelType = 'geapck'` adds response gradients to the PCK system.
- `PCKType` is required for `apck` and `geapck`.
- `GradientMode` applies only to `geapck`.
- `aPC_ComputeUQ` and `aPC_ComputeSensitivity` currently support `apc` only.

## Gradient Rules
- For provided gradients, require `data.dYdX_train` with shape `N x M`.
- For finite-difference gradients, require `data.ModelFcn`.
- GEAPCK internally maps gradients from physical `X` space to standardized `U` space and then to reduced model space.
- Supported reduction methods for GE gradient propagation currently include linear-map methods such as `pls`, `pca`, `spca_hsic`, `rp`, `mcp`, and `ica`.
- Nonlinear kernel reductions like `kpca` and `skpca_hsic` are not currently suitable for GE gradient propagation.

## Kriging and PCK Rules
- APCK supports `gaussian`, `exponential`, and `matern32` correlation functions.
- GEAPCK supports only differentiable kernels currently implemented for gradient blocks, namely `gaussian` and `matern32`.
- `exponential` is intentionally not used in GEAPCK because it is not differentiable at zero.
- `theta` optimization is numerical and may show small serial/parallel differences when `optimize_theta = true`.
- Small numerical differences between serial and parallel optimal PCK runs are acceptable if model selection logic is unchanged.

## Parallelism Rules
- Respect `ModelOpts.Parallel`; do not assume one parallel strategy fits all models.
- APC and sequential PCK typically parallelize over the outer configuration grid.
- Optimal APCK/GEAPCK may use trend-level parallelism when only one outer configuration is being evaluated.
- Keep worker outputs deterministic; collect progress with `parallel.pool.DataQueue`.
- Avoid worker-side noisy printing when a DataQueue-based progress display is active.

## Error Handling
- Validate required fields explicitly with `isfield` and `isempty` checks.
- Use `error(...)` for invalid configuration or unsupported combinations.
- Use `warning(...)` only for recoverable, non-fatal situations.
- Do not swallow exceptions silently.
- If a feature is unsupported for a given model type, fail fast with a clear error message.

## Documentation Rules
- Update `README.md` whenever user-facing behavior, configuration, or workflow changes.
- Keep `AGENTS.md` aligned with actual repository behavior.
- There are no `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` files in this repository at present.

## Safe Change Practices
- Prefer minimal diffs.
- Do not reformat unrelated files.
- Keep APIs backward-compatible where practical.
- Update both training and prediction code when changing saved model state.
- If changing print/report formatting, keep outputs readable and structurally consistent.

## Verification Checklist
- Paths initialize with `aPC`.
- Modified workflows run without manual path hacks.
- Updated functions preserve expected shapes and field names.
- New options have defaults and validation.
- Relevant MATLAB tests pass.
- Example or comparison script still runs when your change touches those paths.
