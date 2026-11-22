# Engine-Owned Processing Steps

Processing engines in `mxproc` have fundamentally different execution granularities: XDS decomposes reduction into eight distinct steps (`initialize` through `report`), DIALS has its own multi-command sequence, while Rugnux executes end-to-end processing in a single binary invocation. Previously, `mxproc` forced every engine onto a shared global `StepType` enum and nine abstract methods on `Analysis`, causing monolithic engines like Rugnux to fabricate fake intermediate steps and partially implemented engines like DIALS to carry empty stub methods.

We decided to remove the global `StepType` enum and fixed workflow transition map, and instead have each engine declare its own ordered `Step` sequence per supported `Workflow` (`PROCESS`, `SCREEN`) as class-level metadata on `Analysis`.

## Considered Options

- **Shared `StepType` superset enum with optional/skipped steps**: Rejected because every new engine with distinct granularity (e.g., DIALS `refine_bravais_settings`, Rugnux single-pass `process`) would either bloat the global enum or force awkward shoehorning into unrelated step names.
- **Engine-owned workflows and steps**: Rejected for now because `PROCESS` vs `SCREEN` remains a meaningful user-level intent shared across engines; engines can simply omit workflows they do not support.

## Consequences

- `Analysis` becomes a thin, engine-agnostic runner and checkpointer with zero hardcoded crystallography step names.
- Checkpoints (`.meta`) record the owning `engine`, `workflow`, and `step` slug, preventing accidental cross-engine resume and allowing CLI resume to auto-detect the engine when omitted.
- `mxproc.reporting` owns only `ReportPayload`, report formatting, and `save_report`; each engine is responsible for assembling its report data from its own step results.
- Legacy step-specific CLI scripts (`auto.spots`, `auto.index`, etc.) resolve by step slug against the chosen engine and fail cleanly with the engine's available steps when unsupported, while `auto.process` gains `--step` and `--list-steps` for arbitrary engine steps.
