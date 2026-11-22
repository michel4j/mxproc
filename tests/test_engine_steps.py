import gzip
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from mxio import XYPair

from mxproc import Analysis, Application, Step, peek_checkpoint_engine, reporting
from mxproc.common import InvalidAnalysisStep, Result, StateType, Workflow
from mxproc.engines.dials import DIALSAnalysis
from mxproc.engines.rugnux import RUGNUXAnalysis
from mxproc.engines.xds import XDSAnalysis
from mxproc.engines.xds.report import build_report_payload
from mxproc.xtal import Experiment, Geometry, Lattice
from tests.test_command import FakeRunner


def _make_experiment(directory: Path, name: str = "ds1") -> Experiment:
    return Experiment(
        name=name,
        identifier=f"{name}-1",
        reference=f"{name}_master.h5",
        directory=directory,
        format="NXmx",
        detector="EIGER",
        wavelength=1.0,
        distance=200.0,
        exposure=0.1,
        frames=[(1, 100)],
        template=f"{name}_master.h5",
        glob=f"{name}_master.h5",
        two_theta=0.0,
        delta_angle=0.5,
        pixel_size=XYPair(0.075, 0.075),
        detector_size=XYPair(4150, 4371),
        detector_origin=XYPair(2075.0, 2185.0),
        geometry=Geometry(),
        cutoff_value=65535.0,
        sensor_thickness=0.45,
    )


class CustomTwoStepAnalysis(Analysis):
    prefix = "custom"
    workflows = {
        Workflow.PROCESS: (
            Step("prepare", "Prepare Workspace"),
            Step("reduce", "Reduce Data", method="run_reduce"),
            Step("finish", "Finalize", checkpoint=False),
        ),
        Workflow.SCREEN: (
            Step("prepare", "Prepare Workspace"),
            Step("plan", "Plan Strategy"),
            Step("finish", "Finalize", checkpoint=False),
        ),
    }

    def __init__(self, args, runner=None):
        super().__init__(args, runner=runner)
        self.executed: list[str] = []

    def prepare(self):
        self.executed.append("prepare")
        self.settings["prepared"] = True
        return {}

    def run_reduce(self):
        self.executed.append("reduce")
        return {
            expt.identifier: Result(state=StateType.SUCCESS, details={"value": 42})
            for expt in self.experiments
        }

    def plan(self):
        self.executed.append("plan")
        return {
            expt.identifier: Result(state=StateType.SUCCESS, details={"angle": 90.0})
            for expt in self.experiments
        }

    def finish(self):
        self.executed.append("finish")


class EngineStepContractTests(unittest.TestCase):
    def test_builtin_engines_declare_expected_steps(self):
        self.assertEqual(
            [s.slug for s in XDSAnalysis.get_steps(Workflow.PROCESS)],
            ["initialize", "spots", "index", "integrate", "symmetry", "scale", "export", "report"],
        )
        self.assertEqual(
            [s.slug for s in XDSAnalysis.get_steps(Workflow.SCREEN)],
            ["initialize", "spots", "index", "strategy", "report"],
        )
        self.assertEqual(
            [s.slug for s in RUGNUXAnalysis.get_steps(Workflow.PROCESS)],
            ["initialize", "process", "report"],
        )
        self.assertEqual(
            [s.slug for s in RUGNUXAnalysis.get_steps(Workflow.SCREEN)],
            ["initialize", "process", "report"],
        )
        self.assertEqual(
            [s.slug for s in DIALSAnalysis.get_steps(Workflow.PROCESS)],
            ["initialize", "spots", "index", "integrate"],
        )
        with self.assertRaises(InvalidAnalysisStep) as ctx:
            DIALSAnalysis.get_steps(Workflow.SCREEN)
        self.assertIn("PROCESS", str(ctx.exception))

    def test_dials_can_be_instantiated_without_abstract_stubs(self):
        app = Application("Test DIALS")
        with tempfile.TemporaryDirectory() as tmp:
            args = app.parser.parse_args(["--dials", "-d", tmp])
            engine = app.get_engine(args)
            self.assertIsInstance(engine, DIALSAnalysis)

    def test_custom_engine_runs_checkpoints_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            work_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            app = Application("Custom")
            args = app.parser.parse_args(["-d", str(work_dir)])
            args.images = [str(data_dir / expt.reference)]

            with patch("mxproc.load_multiple", return_value=[expt]):
                engine = CustomTwoStepAnalysis(args, runner=FakeRunner())
                rc = engine.run(single=True)

            self.assertEqual(rc, 0)
            self.assertEqual(engine.executed, ["prepare"])
            # Checkpointed even though prepare returned {}
            self.assertTrue((work_dir / "prepare.meta").exists())
            self.assertEqual(peek_checkpoint_engine(work_dir), "custom")

            # Resume from second step to completion
            resume_args = app.parser.parse_args(["-d", str(work_dir)])
            resumed = CustomTwoStepAnalysis(resume_args, runner=FakeRunner())
            rc = resumed.run(step="reduce")
            self.assertEqual(rc, 0)
            self.assertEqual(resumed.executed, ["reduce", "finish"])
            self.assertTrue(resumed.settings["prepared"])
            self.assertEqual(resumed.get_step_result(expt, "reduce").get("value"), 42)
            self.assertTrue((work_dir / "reduce.meta").exists())
            self.assertFalse((work_dir / "finish.meta").exists())

    def test_resume_restores_screen_workflow_and_allows_explicit_process_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            work_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            app = Application("Custom")
            args = app.parser.parse_args(["-s", "-d", str(work_dir)])
            args.images = [str(data_dir / expt.reference)]

            with patch("mxproc.load_multiple", return_value=[expt]):
                engine = CustomTwoStepAnalysis(args, runner=FakeRunner())
                self.assertEqual(engine.run(single=True), 0)

            # Resume 'plan' without passing -s: restored SCREEN workflow should accept it
            resume_args = app.parser.parse_args(["-d", str(work_dir)])
            resumed_screen = CustomTwoStepAnalysis(resume_args, runner=FakeRunner())
            self.assertEqual(resumed_screen.run(step="plan"), 0)
            self.assertEqual(resumed_screen.workflow, Workflow.SCREEN)
            self.assertEqual(resumed_screen.executed, ["plan", "finish"])

            # Resume 'reduce' (PROCESS-only) from a SCREEN checkpoint without -s: stays PROCESS
            resumed_proc = CustomTwoStepAnalysis(resume_args, runner=FakeRunner())
            self.assertEqual(resumed_proc.run(step="reduce"), 0)
            self.assertEqual(resumed_proc.workflow, Workflow.PROCESS)
            self.assertEqual(resumed_proc.executed, ["reduce", "finish"])

    def test_unsupported_step_fails_before_checkpoint_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Application("Test Rugnux")
            args = app.parser.parse_args(["--rugnux", "-d", tmp])
            engine = RUGNUXAnalysis(args, runner=FakeRunner())
            with self.assertRaises(InvalidAnalysisStep) as ctx:
                engine.run(step="index")
            msg = str(ctx.exception)
            self.assertIn("index", msg)
            self.assertIn("initialize, process, report", msg)

    def test_checkpoint_rejects_engine_mismatch_and_legacy_missing_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            work_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            app = Application("Test")
            rugnux_args = app.parser.parse_args(["--rugnux", "-d", str(work_dir)])
            rugnux_args.images = [str(data_dir / expt.reference)]
            with patch("mxproc.load_multiple", return_value=[expt]):
                self.assertEqual(RUGNUXAnalysis(rugnux_args, runner=FakeRunner()).run(single=True), 0)

            # Explicit XDS resume against rugnux checkpoint must fail
            xds_args = app.parser.parse_args(["--xds", "-d", str(work_dir)])
            xds_engine = XDSAnalysis(xds_args, runner=FakeRunner())
            with self.assertRaises(InvalidAnalysisStep) as ctx:
                xds_engine.run(step="spots")
            self.assertIn("rugnux", str(ctx.exception))
            self.assertIn("xds", str(ctx.exception))

            # Legacy checkpoint without 'engine' key must fail
            legacy_meta = {
                "options": xds_engine.options,
                "experiments": [expt],
                "results": {},
                "settings": {},
            }
            with gzip.open(work_dir / "latest.meta", "wb") as handle:
                yaml.dump(legacy_meta, handle, encoding="utf-8")
            with self.assertRaises(InvalidAnalysisStep) as ctx:
                xds_engine.load()
            self.assertIn("missing engine metadata", str(ctx.exception))

    def test_multi_option_sets_merge_false_and_preserves_across_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            work_merged = tmp_path / "merged"
            work_separate = tmp_path / "separate"
            data_dir.mkdir()
            expt1 = _make_experiment(data_dir, "ds1")
            expt2 = _make_experiment(data_dir, "ds2")
            images = [str(data_dir / expt1.reference), str(data_dir / expt2.reference)]

            app = Application("Multi Test")

            # Single experiment: merge=False, multi=False
            single_args = app.parser.parse_args(["-d", str(tmp_path / "single")])
            single_args.images = [images[0]]
            with patch("mxproc.load_multiple", return_value=[expt1]):
                single_engine = CustomTwoStepAnalysis(single_args, runner=FakeRunner())
            self.assertFalse(single_engine.options.merge)
            self.assertFalse(single_engine.options.multi)

            # Multi experiment default: merge=True, multi=False
            merged_args = app.parser.parse_args(["-d", str(work_merged)])
            merged_args.images = images
            with patch("mxproc.load_multiple", return_value=[expt1, expt2]):
                merged_engine = CustomTwoStepAnalysis(merged_args, runner=FakeRunner())
                self.assertTrue(merged_engine.options.merge)
                self.assertFalse(merged_engine.options.multi)
                self.assertEqual(merged_engine.run(single=True), 0)

            # Resuming merged run with -m switches to merge=False, multi=True
            override_args = app.parser.parse_args(["-m", "-d", str(work_merged)])
            override_engine = CustomTwoStepAnalysis(override_args, runner=FakeRunner())
            override_engine.load()
            self.assertFalse(override_engine.options.merge)
            self.assertTrue(override_engine.options.multi)

            # Multi experiment with -m: merge=False, multi=True
            sep_args = app.parser.parse_args(["-m", "-d", str(work_separate)])
            sep_args.images = images
            with patch("mxproc.load_multiple", return_value=[expt1, expt2]):
                sep_engine = CustomTwoStepAnalysis(sep_args, runner=FakeRunner())
                self.assertFalse(sep_engine.options.merge)
                self.assertTrue(sep_engine.options.multi)
                self.assertEqual(sep_engine.run(single=True), 0)

            # Resuming -m run without re-passing -m preserves merge=False, multi=True
            resume_args = app.parser.parse_args(["-d", str(work_separate)])
            resumed = CustomTwoStepAnalysis(resume_args, runner=FakeRunner())
            resumed.load()
            self.assertFalse(resumed.options.merge)
            self.assertTrue(resumed.options.multi)


class CLIStepTests(unittest.TestCase):
    def test_auto_detects_engine_on_resume_when_flag_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            work_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            app = Application("Auto")
            args = app.parser.parse_args(["--rugnux", "-d", str(work_dir)])
            args.images = [str(data_dir / expt.reference)]
            with patch("mxproc.load_multiple", return_value=[expt]):
                self.assertEqual(RUGNUXAnalysis(args, runner=FakeRunner()).run(single=True), 0)

            resume_args = app.parser.parse_args(["-d", str(work_dir)])
            self.assertEqual(Application.resolve_engine_name(resume_args), "RUGNUX")
            self.assertIsInstance(Application.get_engine(resume_args), RUGNUXAnalysis)

            explicit_args = app.parser.parse_args(["--xds", "-d", str(work_dir)])
            self.assertEqual(Application.resolve_engine_name(explicit_args), "XDS")

            fresh_args = app.parser.parse_args([str(data_dir / expt.reference)])
            self.assertEqual(Application.resolve_engine_name(fresh_args), "XDS")

    def test_list_steps_prints_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "should_not_exist"
            app = Application("List Steps")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = app.run(argv=["--rugnux", "--list-steps", "-d", str(target_dir)])
            self.assertEqual(rc, 0)
            output = buf.getvalue()
            self.assertIn("Engine: rugnux", output)
            self.assertIn("initialize", output)
            self.assertIn("process", output)
            self.assertIn("report", output)
            self.assertFalse(target_dir.exists())

    def test_cli_returns_nonzero_on_unsupported_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Application("Legacy index", step="index")
            rc = app.run(argv=["--rugnux", "-d", tmp])
            self.assertEqual(rc, 1)

    def test_strategy_entrypoint_defaults_to_screen_workflow(self):
        app = Application("Screen and Determine Strategy", step="strategy", screen=True)
        args = app.parser.parse_args([])
        self.assertEqual(args.step, "strategy")
        self.assertTrue(args.screen)


class XDSReportSeamTests(unittest.TestCase):
    def test_reporting_package_has_no_build_report_payload(self):
        self.assertFalse(hasattr(reporting, "build_report_payload"))

    def test_xds_build_report_payload_reads_slug_keyed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            expt = _make_experiment(work_dir)
            app = Application("XDS Report Test")
            args = app.parser.parse_args(["--xds", "-d", str(work_dir)])
            analysis = XDSAnalysis(args, runner=FakeRunner())
            analysis.experiments = [expt]
            analysis.options.working_directories = {expt.identifier: work_dir}
            analysis.settings = {
                "symmetry_method": "automatically assigned",
                "resolution_method": "I/Sigma(I) > 1.0",
                "reflection_files": {"mtz": "final.mtz"},
            }
            lat = Lattice(19, 50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
            analysis.results[expt.identifier] = {
                "index": Result(details={"lattice": lat, "symmetry_lattice": lat, "point_groups": ("222",)}),
                "integrate": Result(details={"lattices": [], "quality": {"errors": []}}),
                "symmetry": Result(
                    details={
                        "lattice": lat,
                        "symmetry": {"candidates": [{"name": "P 21 21 21", "number": 19, "probability": 0.99}]},
                        "quality": {"summary": {"score": 0.9, "resolution": 1.8}, "frames": [], "differences": []},
                    }
                ),
                "scale": Result(
                    details={
                        "lattice": lat,
                        "quality": {
                            "summary": {"score": 0.92, "resolution": 1.75, "completeness": 99.1},
                            "statistics": [],
                            "twinning": {},
                        },
                    }
                ),
            }

            payload = build_report_payload(analysis)
            self.assertEqual(len(payload.experiments), 1)
            expt_data = payload.experiments[0]
            self.assertIsNotNone(expt_data.indexing)
            self.assertIsNotNone(expt_data.integration)
            self.assertIsNotNone(expt_data.symmetry)
            self.assertIsNotNone(expt_data.scaling)
            self.assertAlmostEqual(expt_data.scaling.summary.score, 0.92)
            self.assertEqual(payload.outputs, {"mtz": "final.mtz"})


if __name__ == "__main__":
    unittest.main()
