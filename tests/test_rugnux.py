import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mxio import XYPair
from mxproc import Application
from mxproc.common import StateType, Workflow
from mxproc.engines.rugnux import RUGNUXAnalysis
from mxproc.engines.rugnux.parser import (
    build_rugnux_command,
    parse_plot_text,
    parse_report_text,
    resolve_rugnux_binary,
)
from mxproc.engines.rugnux.scoring import compute_score
from mxproc.xtal import Experiment, Geometry
from tests.test_command import FakeRunner


SAMPLE_REPORT = """\
# ******************************************************************************
#  RUGNUX PROCESSING REPORT
# ******************************************************************************
REPORT_VERSION= 8
RUGNUX_VERSION= 1.0.0-rc.186
INPUT_FILE= /data/xtal/insulin/insulin_001.img
OUTPUT_PREFIX= insul

# ******************************************************************************
#  SUMMARY
# ******************************************************************************
VERDICT= WARNINGS
VERDICT_TEXT= Merged to 1.49 A in I 2 3.
WARNING_COUNT= 1
PATHOLOGY_FLAGS= OVERLOADS
ADVICE= NONE
# WARNING: Overloads: 241 integrated reflections held a saturated pixel.

# ******************************************************************************
#  1. DATA SET AND GEOMETRY
# ******************************************************************************
EXPERIMENT_TYPE= ROTATION
IMAGES_PROCESSED= 120
WAVELENGTH= 0.97934
DETECTOR_DISTANCE= 200.367
BEAM_CENTRE= 1515.69 1536.70

# ******************************************************************************
#  2. CRYSTAL
# ******************************************************************************
LATTICE_FOUND= TRUE
UNIT_CELL_CONSTANTS= 78.175 78.175 78.175 90.000 90.000 90.000
MOSAICITY_DEG= 0.109
SPACE_GROUP_NUMBER= 197
SPACE_GROUP_NAME= I 2 3
SPACE_GROUP_ALTERNATIVES= I 21 3
SPACE_GROUP_ACHIRAL_ALTERNATIVES= I m -3

# ******************************************************************************
#  3. MERGED DATA
# ******************************************************************************
MERGE= PERFORMED
INCLUDE_RESOLUTION_RANGE= 27.639 1.486
FITTED_RESOLUTION= 1.52
FRIEDELS_LAW= TRUE
UNIQUE_REFLECTIONS= 10791
TOTAL_OBSERVATIONS= 60517
COMPLETENESS= 81.1
MULTIPLICITY= 5.61
I_OVER_SIGMA= 19.09
R_MEAS= 0.0383
R_MERGE= 0.0353
R_PIM= 0.0145
CC_HALF= 0.9992
CC_STAR= 0.9998
SIGANO= 0.590
CC_ANOM= -0.0172
WILSON_B= 34.37
ISA= undetermined

#     d_min    N_obs   N_uniq N_possib    Compl    Mult  <I/sig>  R_merge   R_meas    R_pim    CC1/2      CC*    CCref   CCanom   SigAno
#  -------- -------- -------- -------- -------- ------- -------- -------- -------- -------- -------- -------- -------- -------- --------
#      4.45     3507      529      541    97.8%     6.6     53.8     3.2%     3.5%     1.4%    99.8%    99.9%        -   -15.1%     0.86
#      1.57     2968     1321     1978    66.8%     2.2      1.1    33.8%    42.7%    25.8%    82.9%    95.2%        -    20.0%     0.32
#  -------- -------- -------- -------- -------- ------- -------- -------- -------- -------- -------- -------- -------- -------- --------
#   Overall    60517    10791    13306    81.1%     5.6     19.1     3.5%     3.8%     1.5%    99.9%   100.0%        -    -1.7%     0.59
#  Wilson B-factor estimate: 34.37 A^2 (correlation 0.997, fitted over 4.00-1.49 A)

ANOM_UNIQUE_REFLECTIONS= 19529
ANOM_TOTAL_OBSERVATIONS= 60517
ANOM_COMPLETENESS= 76.5
"""

SAMPLE_PLOT = """\
#image  angle_deg          bkg resolution_A    spots      scale_G   sigma_M_deg  cc_to_merge     cc_n  merged disposition
     0   100.2500        158.3        8.192      182       1.0206        0.1047       0.9656     1162       1           1
     1   100.7500        158.5        7.411      196       1.0203        0.1076       0.9272     1181       1           1
"""


def _make_experiment(directory: Path, name: str = "ds1", ref: str = "ds1_master.h5") -> Experiment:
    return Experiment(
        name=name,
        identifier=f"{name}-1",
        reference=ref,
        directory=directory,
        format="NXmx",
        detector="EIGER",
        wavelength=0.97934,
        distance=200.0,
        exposure=0.1,
        frames=[(1, 120)],
        template=ref,
        glob=ref,
        two_theta=0.0,
        delta_angle=0.5,
        pixel_size=XYPair(0.075, 0.075),
        detector_size=XYPair(4150, 4371),
        detector_origin=XYPair(2075.0, 2185.0),
        geometry=Geometry(),
        cutoff_value=65535.0,
        sensor_thickness=0.45,
    )


class RugnuxParserTests(unittest.TestCase):
    def test_resolve_rugnux_binary_prefers_env_and_falls_back_to_default(self):
        with patch.dict(os.environ, {"RUGNUX_BIN": "/custom/rugnux"}, clear=False):
            self.assertEqual(resolve_rugnux_binary(), "/custom/rugnux")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUGNUX_BIN", None)
            with patch("shutil.which", return_value="/usr/local/bin/rugnux"):
                self.assertEqual(resolve_rugnux_binary(), "/usr/local/bin/rugnux")
            with patch("shutil.which", return_value=None):
                self.assertEqual(resolve_rugnux_binary(), "rugnux")

    def test_build_rugnux_command_formats_options(self):
        cmd = build_rugnux_command(
            [Path("/data/a_master.h5"), Path("/data/b_master.h5")],
            prefix="rugnux",
            anomalous=True,
            spacegroup=19,
            resolution=1.8,
            beam_center=(1024.5, 1025.5),
            frames=[(10, 100)],
            no_merge=True,
            binary="/opt/rugnux/bin/rugnux",
        )
        self.assertIn("/opt/rugnux/bin/rugnux -o rugnux", cmd)
        self.assertIn(" -A ", cmd)
        self.assertIn(" -S 19 ", cmd)
        self.assertIn(" --scaling-high-resolution 1.8 ", cmd)
        self.assertIn(" --beam-x 1024.5 --beam-y 1025.5 ", cmd)
        self.assertIn(" -s 9 -e 100 ", cmd)
        self.assertIn(" --no-merge ", cmd)
        self.assertIn("/data/a_master.h5 /data/b_master.h5", cmd)

    def test_parse_report_and_plot(self):
        report = parse_report_text(SAMPLE_REPORT)
        self.assertEqual(report.verdict, "WARNINGS")
        self.assertTrue(report.is_merged)
        self.assertEqual(report.get_int("SPACE_GROUP_NUMBER"), 197)
        self.assertAlmostEqual(report.get_float("FITTED_RESOLUTION"), 1.52)
        self.assertIsNone(report.get_float("ISA"))
        self.assertEqual(len(report.primary_shells), 2)
        self.assertAlmostEqual(report.primary_shells[0].cc_half, 99.8)
        self.assertEqual(report.pathology_flags, ["OVERLOADS"])

        score = compute_score(report)
        self.assertGreater(score, 0.8)
        self.assertLessEqual(score, 1.0)

        plot = parse_plot_text(SAMPLE_PLOT)
        self.assertEqual(plot.n_frames, 2)
        self.assertEqual(plot.frame_numbers, [1, 2])
        self.assertEqual(plot.get_column("spots"), [182, 196])


class RugnuxAnalysisTests(unittest.TestCase):
    def test_application_selects_rugnux_engine(self):
        app = Application("Test Rugnux")
        with tempfile.TemporaryDirectory() as tmp:
            args = app.parser.parse_args(["--rugnux", "-d", tmp])
            engine = app.get_engine(args)
            self.assertIsInstance(engine, RUGNUXAnalysis)

    def test_full_process_workflow_with_fake_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            def on_run(cmd, work_dir):
                (work_dir / "rugnux_report.txt").write_text(SAMPLE_REPORT, encoding="utf-8")
                (work_dir / "rugnux_plot.txt").write_text(SAMPLE_PLOT, encoding="utf-8")
                (work_dir / "rugnux.mtz").write_text("mtz", encoding="utf-8")
                (work_dir / "rugnux.cif").write_text("cif", encoding="utf-8")
                (work_dir / "rugnux.hkl").write_text("hkl", encoding="utf-8")

            runner = FakeRunner(on_run=on_run)
            app = Application("Test Rugnux")
            args = app.parser.parse_args(["--rugnux", "-d", str(out_dir)])

            with patch("mxproc.load_multiple", return_value=[expt]):
                args.images = [str(data_dir / expt.reference)]
                analysis = RUGNUXAnalysis(args, runner=runner)
                exit_code = analysis.run()

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertTrue((out_dir / "initialize.meta").exists())
            self.assertTrue((out_dir / "process.meta").exists())
            self.assertFalse((out_dir / "report.meta").exists())
            self.assertTrue((out_dir / "report.html").exists())
            self.assertTrue((out_dir / "report.json").exists())
            self.assertTrue((out_dir / "report.txt").exists())

            report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            titles = [sec["title"] for sec in report_json["details"]]
            self.assertIn("Data Quality Summary", titles)
            self.assertIn("Statistics of Intensities by Frame Number", titles)
            self.assertIn("Statistics of Final Reflections by Shell", titles)
            self.assertIn("Diagnostics & Warnings", titles)

            process_res = analysis.get_step_result(expt, "process")
            self.assertIsNotNone(process_res)
            self.assertEqual(process_res.state, StateType.WARNING)
            self.assertEqual(process_res.get("lattice").spacegroup, 197)
            self.assertAlmostEqual(process_res.get("quality.summary.r_meas"), 3.83)
            self.assertAlmostEqual(process_res.get("quality.summary.cc_half"), 99.92)
            self.assertEqual(analysis.settings["reflection_files"]["mtz"], "rugnux.mtz")

    def test_multi_sweep_merge_runs_rugnux_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "work"
            data_dir.mkdir()
            expt1 = _make_experiment(data_dir, "ds1", "ds1_master.h5")
            expt2 = _make_experiment(data_dir, "ds2", "ds2_master.h5")

            def on_run(cmd, work_dir):
                (work_dir / "rugnux_report.txt").write_text(SAMPLE_REPORT, encoding="utf-8")

            runner = FakeRunner(on_run=on_run)
            app = Application("Test Rugnux")
            args = app.parser.parse_args(["--rugnux", "-d", str(out_dir)])
            args.images = [str(data_dir / expt1.reference), str(data_dir / expt2.reference)]

            with patch("mxproc.load_multiple", return_value=[expt1, expt2]):
                analysis = RUGNUXAnalysis(args, runner=runner)
                exit_code = analysis.run()

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runner.calls), 1)
            cmd, work_dir, _, _ = runner.calls[0]
            self.assertEqual(work_dir, out_dir)
            self.assertIn("ds1_master.h5", cmd)
            self.assertIn("ds2_master.h5", cmd)

    def test_multi_separate_runs_rugnux_per_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "work"
            data_dir.mkdir()
            expt1 = _make_experiment(data_dir, "ds1", "ds1_master.h5")
            expt2 = _make_experiment(data_dir, "ds2", "ds2_master.h5")

            def on_run(cmd, work_dir):
                (work_dir / "rugnux_report.txt").write_text(SAMPLE_REPORT, encoding="utf-8")
                (work_dir / "rugnux.mtz").write_text("mtz", encoding="utf-8")

            runner = FakeRunner(on_run=on_run)
            app = Application("Test Rugnux")
            args = app.parser.parse_args(["--rugnux", "-m", "-d", str(out_dir)])
            args.images = [str(data_dir / expt1.reference), str(data_dir / expt2.reference)]

            with patch("mxproc.load_multiple", return_value=[expt1, expt2]):
                analysis = RUGNUXAnalysis(args, runner=runner)
                exit_code = analysis.run()

            self.assertEqual(exit_code, 0)
            self.assertFalse(analysis.options.merge)
            self.assertTrue(analysis.options.multi)
            self.assertEqual(len(runner.calls), 2)
            self.assertEqual({call[1] for call in runner.calls}, {out_dir / "ds1", out_dir / "ds2"})
            self.assertTrue((out_dir / "ds1" / "report.html").exists())
            self.assertTrue((out_dir / "ds2" / "report.html").exists())
            ds1_report = json.loads((out_dir / "ds1" / "report.json").read_text(encoding="utf-8"))
            ds2_report = json.loads((out_dir / "ds2" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(ds1_report["outputs"]["mtz"], "ds1/rugnux.mtz")
            self.assertEqual(ds2_report["outputs"]["mtz"], "ds2/rugnux.mtz")

    def test_screen_workflow_passes_no_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            def on_run(cmd, work_dir):
                (work_dir / "rugnux_report.txt").write_text(SAMPLE_REPORT, encoding="utf-8")

            runner = FakeRunner(on_run=on_run)
            app = Application("Test Rugnux")
            args = app.parser.parse_args(["--rugnux", "-s", "-d", str(out_dir)])
            args.images = [str(data_dir / expt.reference)]

            with patch("mxproc.load_multiple", return_value=[expt]):
                analysis = RUGNUXAnalysis(args, runner=runner)
                exit_code = analysis.run()

            self.assertEqual(exit_code, 0)
            self.assertEqual(analysis.workflow, Workflow.SCREEN)
            self.assertEqual(len(runner.calls), 1)
            self.assertIn("--no-merge", runner.calls[0][0])

    def test_resume_report_step_reads_disk_without_rerunning_rugnux(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "work"
            data_dir.mkdir()
            expt = _make_experiment(data_dir)

            def on_run(cmd, work_dir):
                (work_dir / "rugnux_report.txt").write_text(SAMPLE_REPORT, encoding="utf-8")
                (work_dir / "rugnux_plot.txt").write_text(SAMPLE_PLOT, encoding="utf-8")

            runner1 = FakeRunner(on_run=on_run)
            app = Application("Test Rugnux")
            args1 = app.parser.parse_args(["--rugnux", "-d", str(out_dir)])
            args1.images = [str(data_dir / expt.reference)]
            with patch("mxproc.load_multiple", return_value=[expt]):
                self.assertEqual(RUGNUXAnalysis(args1, runner=runner1).run(), 0)

            (out_dir / "report.html").unlink()
            runner2 = FakeRunner()
            args2 = app.parser.parse_args(["--rugnux", "-d", str(out_dir)])
            exit_code = RUGNUXAnalysis(args2, runner=runner2).run(step="report")
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runner2.calls), 0)
            self.assertTrue((out_dir / "report.html").exists())

            (out_dir / "rugnux_report.txt").unlink()
            runner3 = FakeRunner()
            exit_code = RUGNUXAnalysis(args2, runner=runner3).run(step="report")
            self.assertEqual(exit_code, 1)
            self.assertEqual(len(runner3.calls), 0)


if __name__ == "__main__":
    unittest.main()
