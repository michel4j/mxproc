from __future__ import annotations

import argparse
from pathlib import Path

from mxproc import Application
from mxproc.command import LocalRunner, SlurmRunner
from mxproc.common import StepType


def _run_application(description: str, step: StepType | None = None, single: bool = False) -> int:
    """
    Helper to instantiate and run an Application. Uses `sys.argv` for CLI arguments.

    :param description: description passed to Application
    :param step: optional StepType to initialize Application with. If None use default.
    :param single: True to run only a single step, False to run all subsequent steps
    :return: exit code from Application.run()
    """
    if step is None:
        app = Application(description=description)
    else:
        app = Application(description=description, step=step)
    return app.run(single=single)


# entrypoints corresponding to files in bin/

def auto_index() -> int:
    """Run the auto.index console entrypoint using sys.argv."""
    return _run_application("Auto-Index Dataset", StepType.INDEX)


def auto_init() -> int:
    """Run the auto.init console entrypoint using sys.argv."""
    return _run_application('Initialize a dataset processing', StepType.INITIALIZE, single=True)


def auto_integrate() -> int:
    """Run the auto.integrate console entrypoint using sys.argv."""
    return _run_application("Integrate Datasets", StepType.INTEGRATE)


def auto_process() -> int:
    """Run the auto.process console entrypoint using sys.argv."""
    return _run_application('Automatically Process a dataset', None)


def auto_scale() -> int:
    """Run the auto.scale console entrypoint using sys.argv."""
    return _run_application("Scale Datasets", StepType.SCALE)


def auto_spots() -> int:
    """Run the auto.spots console entrypoint using sys.argv."""
    return _run_application("Find Strong Spots", StepType.SPOTS, single=True)


def auto_strategy() -> int:
    """Run the auto.strategy console entrypoint using sys.argv."""
    return _run_application("Screen and Determine Strategy", StepType.STRATEGY)


def auto_symmetry() -> int:
    """Run the auto.symmetry console entrypoint using sys.argv."""
    return _run_application("Determine and Apply Symmetry", StepType.SYMMETRY)


def auto_xds() -> int:
    """
    Run the functionality previously provided by the standalone `bin/auto.xds` script.

    This function parses `sys.argv` just like the original script and either submits
    a SLURM job via SSH or runs the command locally.
    """
    parser = argparse.ArgumentParser(description="Generate and submit a SLURM job script.")
    parser.add_argument("command", nargs='?', type=str, default="xds_par", help="Command to run")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--cpus", type=int, help="Number of CPUs per task")
    parser.add_argument("--tasks", type=int, help="Total Number of tasks per node")
    parser.add_argument("--host", type=str, help="Slurm host or user@host for SSH")
    parser.add_argument("--partition", type=str, default='batch', help="Slurm partition")

    args = parser.parse_args()
    if args.nodes and args.cpus and args.tasks and args.host:
        runner = SlurmRunner(
            nodes=args.nodes,
            cpus=args.cpus,
            tasks=args.tasks,
            host=args.host,
            partition=args.partition,
        )
        runner.run(args.command, Path.cwd(), cluster=True, logfile=None, show_progress=False)
    else:
        LocalRunner().run(args.command, Path.cwd(), logfile=None, show_progress=False)

    return 0


