import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy

from mxio import XYPair
from mxproc.common import StateType
from mxproc.engines.xds.indexing import (
    IndexParamManager,
    IndexProblem,
    auto_index,
    diagnose_indexing,
)
from mxproc.xtal import Experiment, Geometry
from tests.test_command import FakeRunner


def _base_log_details(**overrides):
    data = {
        "lattices": [{"character": "oP", "unit_cell": (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)}],
        "quality": {"pixel_error": 1.0},
        "spots": {"total": 100, "misfits": 5},
        "subtrees": [{"population": 100}],
        "exp_ang_err": 0.5,
        "exp_pos_err": 2.0,
        "max_integral_dev": 0.05,
    }
    data.update(overrides)
    return data


def _base_param_details():
    return {"sg_number": 19, "unit_cell": (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)}


class DiagnoseIndexingTests(unittest.TestCase):
    def test_clean_indexing_succeeds(self):
        result = diagnose_indexing(_base_log_details(), _base_param_details())
        self.assertEqual(result.state, StateType.SUCCESS)
        self.assertEqual(result.flags, IndexProblem.NONE)
        self.assertEqual(result.get("symmetry_lattice").spacegroup, 16)

    def test_detects_inverted_axis(self):
        log = _base_log_details(
            spots={"total": 100, "misfits": 80},
            cluster_indices=[{"hkl": [1.35, 2.0, 3.0]}],
            quality={"pixel_error": 1.5},
        )
        result = diagnose_indexing(log, _base_param_details())
        self.assertIn(IndexProblem.INVERTED_AXIS, IndexProblem(result.flags))
        self.assertEqual(result.state, StateType.WARNING)

    def test_detects_poor_solution(self):
        log = _base_log_details(
            spots={"total": 100, "misfits": 40},
            quality={"pixel_error": 6.0},
        )
        result = diagnose_indexing(log, _base_param_details())
        self.assertIn(IndexProblem.POOR_SOLUTION, IndexProblem(result.flags))
        self.assertEqual(result.state, StateType.WARNING)

    def test_detects_log_failure_message(self):
        log = _base_log_details(message_1="REFINEMENT DID NOT CONVERGE")
        result = diagnose_indexing(log, _base_param_details())
        self.assertIn(IndexProblem.REFINEMENT_FAILED, IndexProblem(result.flags))
        self.assertEqual(result.state, StateType.FAILURE)


class IndexParamManagerTests(unittest.TestCase):
    def test_inverted_axis_retries_once(self):
        mgr = IndexParamManager(data_range=[(1, 10)], spot_range=[(1, 10)])
        retry1, msgs1 = mgr.update_parameters(IndexProblem.INVERTED_AXIS)
        self.assertTrue(retry1)
        self.assertTrue(mgr.get_parameters()["invert_spindle"])
        self.assertEqual(len(msgs1), 1)

        retry2, _ = mgr.update_parameters(IndexProblem.INVERTED_AXIS)
        self.assertFalse(retry2)

    def test_low_fraction_and_poor_solution_updates_spot_range(self):
        mgr = IndexParamManager(data_range=[(1, 100)], spot_range=[(1, 100)])
        retry, _ = mgr.update_parameters(
            IndexProblem.LOW_INDEXED_FRACTION | IndexProblem.POOR_SOLUTION,
            spot_range=(20, 45),
        )
        self.assertTrue(retry)
        self.assertEqual(mgr.get_parameters()["spot_range"], [(20, 45)])

    def test_refinement_failed_exhausts_flags(self):
        mgr = IndexParamManager(data_range=[(1, 10)], spot_range=[(1, 10)])
        self.assertTrue(mgr.update_parameters(IndexProblem.REFINEMENT_FAILED)[0])
        self.assertEqual(mgr.get_parameters()["refine_index"], ("CELL", "BEAM", "ORIENTATION"))
        self.assertTrue(mgr.update_parameters(IndexProblem.REFINEMENT_FAILED)[0])
        self.assertEqual(mgr.get_parameters()["refine_index"], ("CELL", "BEAM"))
        self.assertFalse(mgr.update_parameters(IndexProblem.REFINEMENT_FAILED)[0])


class AutoIndexTests(unittest.TestCase):
    def _dummy_experiment(self, tmp_dir: Path) -> Experiment:
        return Experiment(
            name="test_ds",
            identifier="test_id",
            reference=str(tmp_dir / "img_0001.cbf"),
            directory=tmp_dir,
            format="CBF",
            detector="PILATUS 6M",
            wavelength=1.0,
            distance=200.0,
            exposure=0.1,
            frames=[(1, 10)],
            template="img_????.cbf",
            glob="img_*.cbf",
            two_theta=0.0,
            delta_angle=0.5,
            pixel_size=XYPair(0.172, 0.172),
            detector_size=XYPair(2463, 2527),
            detector_origin=XYPair(1231.5, 1263.5),
            geometry=Geometry(
                detector=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                goniometer=(1.0, 0.0, 0.0),
                beam=(0.0, 0.0, 1.0),
                polarization=(0.99, (0.0, 1.0, 0.0)),
            ),
            cutoff_value=1000000,
            sensor_thickness=0.32,
            start_angle=0.0,
        )

    def test_auto_index_retries_and_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            numpy.save(work_dir / "spots", numpy.array([[100.0, 100.0, 1.0, 10.0, 1, 0, 0]]))
            expt = self._dummy_experiment(work_dir)
            runner = FakeRunner()

            parse_calls = {"n": 0}

            def fake_parse(filename, silent=False):
                if filename.endswith("IDXREF.LP"):
                    parse_calls["n"] += 1
                    if parse_calls["n"] == 1:
                        return _base_log_details(message_1="REFINEMENT DID NOT CONVERGE")
                    return _base_log_details()
                return _base_param_details()

            with (
                patch("mxproc.engines.xds.indexing.XDSParser.parse", side_effect=fake_parse),
                patch("mxproc.engines.xds.indexing.get_spot_distribution", return_value=(None, (1, 10))),
            ):
                result = auto_index(
                    expt,
                    {"data_range": [(1, 10)], "spot_range": [(1, 10)]},
                    work_dir=work_dir,
                    runner=runner,
                )

            self.assertEqual(result.state, StateType.SUCCESS)
            self.assertEqual(len(runner.calls), 2)

    def test_auto_index_stops_on_software_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            numpy.save(work_dir / "spots", numpy.array([[100.0, 100.0, 1.0, 10.0, 1, 0, 0]]))
            expt = self._dummy_experiment(work_dir)
            runner = FakeRunner()

            with patch("mxproc.engines.xds.indexing.XDSParser.parse", side_effect=FileNotFoundError("IDXREF.LP")):
                result = auto_index(
                    expt,
                    {"data_range": [(1, 10)], "spot_range": [(1, 10)]},
                    work_dir=work_dir,
                    runner=runner,
                )

            self.assertEqual(result.state, StateType.FAILURE)
            self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
