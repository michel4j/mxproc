# mxproc

Automated macromolecular crystallography data processing pipeline supporting multiple crystallographic engines.

## Language

**Engine**:
A crystallographic processing integration (such as XDS, Rugnux, or DIALS) that owns its own steps and executes workflows on diffraction datasets.
_Avoid_: backend, processor

**Workflow**:
A high-level processing goal shared across engines (`PROCESS` for full data reduction, `SCREEN` for characterization and strategy). Each engine defines its own ordered step sequence for each workflow it supports.
_Avoid_: mode, pipeline

**Step**:
A single named, ordered operation declared by an engine within a workflow, identified by a slug and optionally persisted as a checkpoint.
_Avoid_: stage, phase, StepType

**Checkpoint**:
A persisted snapshot of an analysis's state (engine, workflow, step, options, experiments, results, and settings) saved after a checkpointed step so later steps can resume without re-running earlier work.
_Avoid_: meta file
