from __future__ import annotations

import argparse
import subprocess
import re
import time
from pathlib import Path

from mxproc import Application
from mxproc.common import StepType


def _run_application(description: str, step: StepType | None = None) -> int:
    """
    Helper to instantiate and run an Application. Uses `sys.argv` for CLI arguments.

    :param description: description passed to Application
    :param step: optional StepType to initialise Application with. If None use default.
    :return: exit code from Application.run()
    """
    if step is None:
        app = Application(description=description)
    else:
        app = Application(description=description, step=step)
    return app.run()


# Entrypoints corresponding to files in bin/

def auto_index() -> int:
    """Run the auto.index console entrypoint using sys.argv."""
    return _run_application("Auto-Index Dataset", StepType.INDEX)


def auto_init() -> int:
    """Run the auto.init console entrypoint using sys.argv."""
    return _run_application('Initialize a dataset processing', StepType.INITIALIZE)


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
    return _run_application("Find Strong Spots", StepType.SPOTS)


def auto_strategy() -> int:
    """Run the auto.strategy console entrypoint using sys.argv."""
    return _run_application("Screen and Determine Strategy", StepType.STRATEGY)


def auto_symmetry() -> int:
    """Run the auto.symmetry console entrypoint using sys.argv."""
    return _run_application("Determine and Apply Symmetry", StepType.SYMMETRY)


# Functions copied from bin/auto.xds to provide the same behaviour when called as a package function

def tail(file_path: Path, sentinel: str = "COMMAND-DONE", interval: float = 0.1):
    """
    Generator function that tails a file and yields new lines as they are added.

    :param file_path: Path to the file to be tailed.
    :param sentinel: return if any line contains this text
    :param interval: Time to wait (in seconds) between checks for new lines.
    """
    # wait for file to exist
    while not file_path.exists():
        print(f'Waiting for {file_path} ...')
        time.sleep(1)

    with open(file_path, 'r') as file:
        # Go to the end of the file
        found = False
        while not found:
            text = file.readline()
            found = bool(re.search(rf'{sentinel}', text))
            yield text


def run_slurm_job(command: str, nodes: int, cpus: int, tasks: int, host: str, partition: str = 'batch', duration: str = '00:30:00'):
    """
    Submit a SLURM job script to the cluster.

    :param command: command to run
    :param nodes: Number of nodes
    :param cpus: Number of CPUs per task
    :param tasks: Total Number of tasks per node
    :param host: hostname or 'user@hostname' for ssh remote submission
    :param duration: maximum duration of slurm job
    :param partition: slurm partition
    """

    # Generate the SLURM job script
    log_file = Path("xds-slurm.log")
    try:
        log_file.unlink()
    except Exception:
        pass

    job_script = (
        "#!/bin/bash \n\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --nodes=1-{nodes}\n"
        f"#SBATCH --ntasks-per-node={tasks//nodes}  # number of MPI processes\n"
        f"#SBATCH --cpus-per-task={cpus}            # number of OpenMP threads\n"
        "#SBATCH --mem=0                            # use all memory\n"
        f"#SBATCH --time={duration}\n"
        f"#SBATCH --output={log_file}\n\n"
        f"# run {command} \n"
        f"{command}\n"
        "echo 'COMMAND-DONE'\n\n"
    )

    # Write the job script to a file
    job_script_file = Path("xds-slurm.sh")
    with open(job_script_file, 'w') as file:
        file.write(job_script)

    current_dir = Path.cwd().resolve()
    ssh_command = f'ssh -x {host} "cd {current_dir}; sbatch {job_script_file}"'
    subprocess.run(ssh_command, shell=True)

    # tail log file until job to complete
    cmd_output = tail(log_file)
    for line in cmd_output:
        print(line, end="")


def run_standard_job(command: str):
    """
    Run a command on the local machine.
    """
    subprocess.run(command, shell=True)


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
        run_slurm_job(
            args.command,
            nodes=args.nodes,
            cpus=args.cpus,
            tasks=args.tasks,
            host=args.host,
            partition=args.partition
        )
    else:
        run_standard_job(args.command)

    return 0
