from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence, Union

from tqdm import tqdm

from mxproc.log import logger


class CommandNotFound(Exception):
    ...


class CommandFailed(Exception):
    ...


def files_exist(files: Sequence[Union[str, Path]], work_dir: Path = Path(".")) -> bool:
    """
    Check if a sequence of files exist on disk relative to work_dir.
    """
    base = Path(work_dir)
    for file in files:
        path = Path(file)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            return False
    return True


class CommandRunner(Protocol):
    def run(
        self,
        cmd: str,
        work_dir: Path = Path("."),
        *,
        desc: str = "",
        logfile: Union[str, Path, None] = "commands.log",
        check_files: Sequence[Union[str, Path]] = (),
        cluster: bool = False,
        job: Any = None,
        show_progress: bool = True,
        capture: bool = False,
    ) -> str:
        ...


class LocalRunner:
    """
    Executes commands locally in an explicit working directory.
    """

    @staticmethod
    def _resolve_log(work_dir: Path, logfile: Union[str, Path, None]) -> Union[Path, None]:
        if logfile is None:
            return None
        path = Path(logfile)
        return path if path.is_absolute() else work_dir / path

    @staticmethod
    async def _exec_with_spinner(cmd: str, work_dir: Path, log_path: Union[Path, None], desc: str) -> int:
        start_str = datetime.now().strftime("%H:%M:%S")
        bar_fmt = "{desc: <83}{elapsed}{postfix}"
        stdout_handle = open(log_path, "a") if log_path is not None else subprocess.DEVNULL
        try:
            with tqdm(desc=f"{start_str} {desc} ... ", miniters=1, leave=False, bar_format=bar_fmt) as spinner:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=work_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stdout_handle,
                )
                while proc.returncode is None:
                    spinner.update()
                    await asyncio.sleep(0.05)
                return proc.returncode
        finally:
            if log_path is not None:
                stdout_handle.close()

    def run(
        self,
        cmd: str,
        work_dir: Path = Path("."),
        *,
        desc: str = "",
        logfile: Union[str, Path, None] = "commands.log",
        check_files: Sequence[Union[str, Path]] = (),
        cluster: bool = False,
        job: Any = None,
        show_progress: bool = True,
        capture: bool = False,
    ) -> str:
        work_dir = Path(work_dir)
        log_path = self._resolve_log(work_dir, logfile)
        stdout_text = ""

        if capture:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            returncode = proc.returncode
            stdout_text = proc.stdout
        elif show_progress:
            try:
                returncode = asyncio.run(self._exec_with_spinner(cmd, work_dir, log_path, desc))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                returncode = loop.run_until_complete(self._exec_with_spinner(cmd, work_dir, log_path, desc))
        else:
            stdout_handle = open(log_path, "a") if log_path is not None else subprocess.DEVNULL
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=work_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stdout_handle,
                )
                returncode = proc.returncode
            finally:
                if log_path is not None:
                    stdout_handle.close()

        with os.scandir(work_dir):  # refresh the file cache
            pass

        if returncode != 0 or not files_exist(check_files, work_dir):
            if desc:
                logger.error_value(desc, "[FAILED]", spacer=" ")
            raise CommandFailed(cmd)

        if desc and not capture:
            logger.info(desc)

        return stdout_text


class SlurmRunner:
    """
    Dispatches cluster commands to SLURM via SSH and delegates local commands to LocalRunner.
    """

    def __init__(
        self,
        nodes: int = 1,
        cpus: int = 1,
        tasks: int = 1,
        host: str = "",
        partition: str = "batch",
        duration: str = "00:30:00",
        local_runner: CommandRunner | None = None,
        poll_interval: float = 0.5,
        squeue_interval: float = 5.0,
        startup_grace: float = 30.0,
    ):
        self.nodes = nodes
        self.cpus = cpus
        self.tasks = tasks
        self.host = host
        self.partition = partition
        self.duration = duration
        self.local_runner = local_runner or LocalRunner()
        self.poll_interval = poll_interval
        self.squeue_interval = squeue_interval
        self.startup_grace = startup_grace

    def _job_in_queue(self, host: str, job_id: str) -> bool:
        proc = subprocess.run(
            f'ssh -x {host} "squeue -h -j {job_id} -o %T"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Transient SSH/squeue error: assume still running unless proven otherwise
            return True
        return bool(proc.stdout.strip())

    def _tail_until_done(self, log_file: Path, host: str, job_id: str | None, out_log: Path | None) -> None:
        sentinel = "COMMAND-DONE"
        start_time = time.monotonic()
        last_squeue_check = start_time

        while not log_file.exists():
            now = time.monotonic()
            if job_id and (now - last_squeue_check) >= self.squeue_interval:
                last_squeue_check = now
                if not self._job_in_queue(host, job_id) and (now - start_time) >= self.startup_grace:
                    raise CommandFailed(f"SLURM job {job_id} ended before creating {log_file.name}")
            elif not job_id and (now - start_time) >= self.startup_grace:
                raise CommandFailed(f"Timed out waiting for SLURM log {log_file.name}")
            time.sleep(self.poll_interval)

        out_handle = open(out_log, "a") if out_log is not None else None
        try:
            with open(log_file, "r") as handle:
                while True:
                    line = handle.readline()
                    if line:
                        if out_handle is not None:
                            out_handle.write(line)
                        if re.search(sentinel, line):
                            return
                        continue

                    now = time.monotonic()
                    if job_id and (now - last_squeue_check) >= self.squeue_interval:
                        last_squeue_check = now
                        if not self._job_in_queue(host, job_id):
                            # Allow brief NFS flush, then drain remaining lines
                            time.sleep(self.poll_interval)
                            for trailing in handle.readlines():
                                if out_handle is not None:
                                    out_handle.write(trailing)
                                if re.search(sentinel, trailing):
                                    return
                            raise CommandFailed(f"SLURM job {job_id} ended without {sentinel}")
                    time.sleep(self.poll_interval)
        finally:
            if out_handle is not None:
                out_handle.close()

    def run(
        self,
        cmd: str,
        work_dir: Path = Path("."),
        *,
        desc: str = "",
        logfile: Union[str, Path, None] = "commands.log",
        check_files: Sequence[Union[str, Path]] = (),
        cluster: bool = False,
        job: Any = None,
        show_progress: bool = True,
        capture: bool = False,
    ) -> str:
        use_slurm = cluster and (job is None or getattr(job, "mode", "SLURM") == "SLURM")
        if not use_slurm:
            return self.local_runner.run(
                cmd,
                work_dir,
                desc=desc,
                logfile=logfile,
                check_files=check_files,
                cluster=False,
                job=job,
                show_progress=show_progress,
                capture=capture,
            )

        work_dir = Path(work_dir)
        nodes = getattr(job, "nodes", self.nodes) if job is not None else self.nodes
        cpus = getattr(job, "cpus", self.cpus) if job is not None else self.cpus
        tasks = getattr(job, "tasks", self.tasks) if job is not None else self.tasks
        host = getattr(job, "host", self.host) if job is not None else self.host
        partition = getattr(job, "partition", self.partition) if job is not None else self.partition

        log_file = work_dir / "xds-slurm.log"
        log_file.unlink(missing_ok=True)

        ntasks_per_node = max(1, tasks // max(1, nodes))
        job_script = (
            "#!/bin/bash\n\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --nodes=1-{nodes}\n"
            f"#SBATCH --ntasks-per-node={ntasks_per_node}  # number of MPI processes\n"
            f"#SBATCH --cpus-per-task={cpus}            # number of OpenMP threads\n"
            "#SBATCH --mem=0                            # use all memory\n"
            f"#SBATCH --time={self.duration}\n"
            f"#SBATCH --output={log_file.name}\n\n"
            f"# run {cmd}\n"
            f"{cmd}\n"
            "echo 'COMMAND-DONE'\n\n"
        )
        script_file = work_dir / "xds-slurm.sh"
        script_file.write_text(job_script)

        ssh_command = f'ssh -x {host} "cd {work_dir.resolve()}; sbatch {script_file.name}"'
        proc = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            if desc:
                logger.error_value(desc, "[FAILED]", spacer=" ")
            raise CommandFailed(f"sbatch failed: {proc.stderr.strip() or proc.stdout.strip()}")

        match = re.search(r"Submitted batch job (\d+)", proc.stdout)
        job_id = match.group(1) if match else None

        out_log = LocalRunner._resolve_log(work_dir, logfile)
        try:
            self._tail_until_done(log_file, host, job_id, out_log)
        except CommandFailed:
            if desc:
                logger.error_value(desc, "[FAILED]", spacer=" ")
            raise

        with os.scandir(work_dir):  # refresh the file cache
            pass
        if not files_exist(check_files, work_dir):
            if desc:
                logger.error_value(desc, "[FAILED]", spacer=" ")
            raise CommandFailed(cmd)

        if desc:
            logger.info(desc)
        return ""
