# mxproc

Automated macromolecular crystallography data processing system supporting multiple crystallographic engines.

## Processing

**Engine**:
A crystallographic processing integration (such as XDS, Rugnux, or DIALS) that owns its own steps and executes workflows on diffraction experiments.
_Avoid_: backend, processor

**Analysis**:
A single execution of an engine over one or more experiments in a working directory, producing results, checkpoints, and reports.
_Avoid_: job, run

**Workflow**:
A high-level processing goal shared across engines: `PROCESS` for full data reduction and `SCREEN` for crystal characterization and collection strategy.
_Avoid_: mode, pipeline

**Step**:
A single named, ordered operation declared by an engine within a workflow.
_Avoid_: stage, phase, StepType

**Checkpoint**:
A saved snapshot of an analysis after a completed step, allowing a workflow to resume from a later step without repeating earlier work.
_Avoid_: meta file

**Auxiliary Program**:
A standalone crystallographic helper tool invoked within an engine's step (such as DOZOR, RADDOSE-3D, POINTLESS, or Xtriage), distinct from a processing engine.
_Avoid_: engine, backend

## Crystallography

**Dataset**:
A raw series of diffraction images on disk; a single dataset may split into multiple experiments when angular gaps divide it into disjoint oscillation ranges.
_Avoid_: sweep

**Experiment**:
A single contiguous diffraction oscillation sweep with uniform detector and goniometer geometry, processed as one unit within an analysis.
_Avoid_: dataset, sweep

**Multi-Experiment Mode**:
Whether an analysis with multiple experiments produces a **merged** combined reflection dataset (default) or **separate** per-experiment outputs (optionally co-scaled by the engine).

**Lattice**:
A crystal's unit cell geometry, Bravais lattice character, and space group assignment.
_Avoid_: unit cell (when symmetry is included)

**Strategy**:
A data acquisition plan produced by a `SCREEN` workflow, recommending oscillation start angle, angular range, maximum frame delta, and radiation-dose-aware exposure settings.

## Outputs

**Quality Score**:
A 0.0–1.0 heuristic score summarizing data or strategy quality for comparing similar analyses; `SCREEN` scores are not comparable to `PROCESS` scores.

**Report**:
The final structured summary bundle presenting quality statistics, plots, diagnostics, and exported reflection files for MxLIVE and AutoProcess consumers.
