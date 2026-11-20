import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mxproc.command import CommandFailed, LocalRunner, SlurmRunner
from mxproc.programs import raddose
from mxproc.xtal import Beam, Lattice


class FakeRunner:
    def __init__(self, stdout: str = "", on_run=None):
        self.calls = []
        self.stdout = stdout
        self.on_run = on_run

    def run(
        self,
        cmd: str,
        work_dir: Path = Path("."),
        *,
        desc: str = "",
        logfile="commands.log",
        check_files=(),
        cluster: bool = False,
        job=None,
        show_progress: bool = True,
        capture: bool = False,
    ) -> str:
        work_dir = Path(work_dir)
        self.calls.append((cmd, work_dir, cluster, job))
        if self.on_run is not None:
            self.on_run(cmd, work_dir)
        return self.stdout


class LocalRunnerTests(unittest.TestCase):
    def test_runs_in_explicit_work_dir_without_changing_cwd(self):
        origin = Path.cwd()
        runner = LocalRunner()
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            runner.run(
                "echo hello > out.txt",
                work_dir,
                desc="write file",
                check_files=("out.txt",),
                show_progress=False,
            )
            self.assertEqual(Path.cwd(), origin)
            self.assertEqual((work_dir / "out.txt").read_text().strip(), "hello")
            self.assertTrue((work_dir / "commands.log").exists())

    def test_raises_when_check_file_missing(self):
        runner = LocalRunner()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CommandFailed):
                runner.run(
                    "true",
                    Path(tmp),
                    check_files=("missing.txt",),
                    show_progress=False,
                )

    def test_raises_on_nonzero_exit(self):
        runner = LocalRunner()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CommandFailed):
                runner.run("false", Path(tmp), show_progress=False)

    def test_capture_returns_stdout(self):
        runner = LocalRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = runner.run("printf 'captured'", Path(tmp), show_progress=False, capture=True)
            self.assertEqual(out, "captured")


class SlurmRunnerTests(unittest.TestCase):
    def test_delegates_non_cluster_to_local_runner(self):
        fake_local = FakeRunner(stdout="local-ok")
        runner = SlurmRunner(host="hpc", local_runner=fake_local)
        with tempfile.TemporaryDirectory() as tmp:
            out = runner.run("echo hi", Path(tmp), cluster=False, capture=True)
            self.assertEqual(out, "local-ok")
            self.assertEqual(len(fake_local.calls), 1)

    def test_delegates_when_job_mode_not_slurm(self):
        fake_local = FakeRunner()
        runner = SlurmRunner(host="hpc", local_runner=fake_local)
        job = SimpleNamespace(mode="LOCAL", nodes=1, cpus=4, tasks=1, host="", partition="batch")
        with tempfile.TemporaryDirectory() as tmp:
            runner.run("xds_par", Path(tmp), cluster=True, job=job)
            self.assertEqual(len(fake_local.calls), 1)

    def test_fails_fast_when_sbatch_fails(self):
        runner = SlurmRunner(nodes=1, cpus=4, tasks=1, host="hpc")
        failed_proc = SimpleNamespace(returncode=1, stdout="", stderr="sbatch: error: invalid partition")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("mxproc.command.subprocess.run", return_value=failed_proc):
                with self.assertRaises(CommandFailed):
                    runner.run("xds_par", Path(tmp), cluster=True)

    def test_fails_without_deadlock_if_job_leaves_queue_before_log_exists(self):
        runner = SlurmRunner(
            nodes=1,
            cpus=4,
            tasks=1,
            host="hpc",
            poll_interval=0.01,
            squeue_interval=0.01,
            startup_grace=0.01,
        )
        sbatch_ok = SimpleNamespace(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
        squeue_empty = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("mxproc.command.subprocess.run", side_effect=[sbatch_ok, squeue_empty]):
                with self.assertRaises(CommandFailed):
                    runner.run("xds_par", Path(tmp), cluster=True)

    def test_fails_without_deadlock_if_job_dies_before_sentinel(self):
        runner = SlurmRunner(
            nodes=1,
            cpus=4,
            tasks=1,
            host="hpc",
            poll_interval=0.01,
            squeue_interval=0.01,
            startup_grace=0.01,
        )
        sbatch_ok = SimpleNamespace(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
        squeue_empty = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            def fake_subproc(cmd, **kwargs):
                if "sbatch" in cmd:
                    (work_dir / "xds-slurm.log").write_text("partial output\n")
                    return sbatch_ok
                return squeue_empty

            with patch("mxproc.command.subprocess.run", side_effect=fake_subproc):
                with self.assertRaises(CommandFailed):
                    runner.run("xds_par", work_dir, cluster=True)

    def test_succeeds_when_sentinel_and_check_files_present(self):
        runner = SlurmRunner(
            nodes=1,
            cpus=4,
            tasks=1,
            host="hpc",
            poll_interval=0.01,
            squeue_interval=0.05,
        )
        sbatch_ok = SimpleNamespace(returncode=0, stdout="Submitted batch job 999\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            def fake_subproc(cmd, **kwargs):
                (work_dir / "xds-slurm.log").write_text("running\nCOMMAND-DONE\n")
                (work_dir / "COLSPOT.LP").write_text("ok\n")
                return sbatch_ok

            with patch("mxproc.command.subprocess.run", side_effect=fake_subproc):
                runner.run("xds_par", work_dir, cluster=True, check_files=("COLSPOT.LP",))
            self.assertIn("xds_par", (work_dir / "xds-slurm.sh").read_text())


class ProgramRunnerTests(unittest.TestCase):
    def test_raddose_get_dose_uses_work_dir_and_runner(self):
        origin = Path.cwd()
        def seed_csv(_cmd, work_dir: Path):
            (work_dir / "RADDOSE.CSV").write_text("Average DWD,Max Dose\n1.5,3.0\n")

        fake = FakeRunner(on_run=seed_csv)
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            ok, details = raddose.get_dose(
                crystal_size=50.0,
                total_range=180.0,
                lattice=Lattice(1, 50.0, 60.0, 70.0, 90.0, 90.0, 90.0),
                beam=Beam(flux=1e12, fwhm_x=50.0, fwhm_y=50.0, aperture=50.0, wavelength=1.0),
                work_dir=work_dir,
                runner=fake,
            )
            self.assertTrue(ok)
            self.assertEqual(Path.cwd(), origin)
            self.assertTrue((work_dir / "RAD.INP").exists())
            self.assertEqual(details["average_dwd"], 1.5)
            self.assertEqual(fake.calls[0][1], work_dir)


if __name__ == "__main__":
    unittest.main()
