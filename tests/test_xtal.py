import unittest

from mxproc.engines.xds.io import find_lattice
from mxproc.xtal import (
    Lattice,
    assess_cell_orientation,
    max_delta_angle,
    score_dataset,
    score_strategy,
)


class XtalGeometryTests(unittest.TestCase):
    def test_assess_cell_orientation_warns_when_long_axis_near_beam(self):
        lattice = Lattice(19, 40.0, 50.0, 200.0, 90.0, 90.0, 90.0)
        near_beam, longest, ratio, warnings = assess_cell_orientation(
            lattice,
            cell_a_axis=(40.0, 0.0, 0.0),
            cell_b_axis=(0.0, 50.0, 0.0),
            cell_c_axis=(0.0, 0.0, 200.0),
            beam_axis=(0.0, 0.0, 1.0),
            rotation_axis=(1.0, 0.0, 0.0),
        )
        self.assertEqual(near_beam["name"], "c")
        self.assertEqual(longest["name"], "c")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(ratio, 0.0)  # c is 0° from beam -> 0 / 90

    def test_assess_cell_orientation_no_warning_when_long_axis_near_spindle(self):
        lattice = Lattice(19, 200.0, 50.0, 40.0, 90.0, 90.0, 90.0)
        near_beam, longest, ratio, warnings = assess_cell_orientation(
            lattice,
            cell_a_axis=(200.0, 0.0, 0.0),
            cell_b_axis=(0.0, 50.0, 0.0),
            cell_c_axis=(0.0, 0.0, 40.0),
            beam_axis=(0.0, 0.0, 1.0),
            rotation_axis=(1.0, 0.0, 0.0),
        )
        self.assertEqual(near_beam["name"], "c")
        self.assertEqual(longest["name"], "a")
        self.assertEqual(warnings, [])
        self.assertEqual(ratio, 0.0)

    def test_max_delta_angle(self):
        delta = max_delta_angle(resolution=2.0, axis_length=100.0, mosaicity=0.2)
        self.assertAlmostEqual(delta, 0.94591559, places=5)


class XtalScoringTests(unittest.TestCase):
    def test_score_strategy_returns_bounded_float(self):
        score = score_strategy(
            completeness=99.0,
            resolution=1.5,
            long_axis_ratio=0.0,
            mosaicity=0.1,
            angle_error=0.2,
            pixel_error=0.8,
            indexed_fraction=0.95,
        )
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_score_dataset_empty_returns_zero(self):
        self.assertEqual(score_dataset({}), 0.0)

    def test_score_dataset_good_metrics(self):
        score = score_dataset({
            "summary": {
                "resolution": 1.6,
                "completeness": 99.0,
                "mosaicity": 0.15,
                "angle_error": 0.2,
                "pixel_error": 0.8,
                "misfits": 5,
                "reflections": 1000,
            },
            "inner_shell": {"r_meas": 3.0, "i_sigma": 40.0},
        })
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)


class FindLatticeTests(unittest.TestCase):
    def test_find_lattice_matches_character(self):
        target = Lattice(19, 50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
        candidates = [
            {
                "character": "mP",
                "unit_cell": (50.0, 60.0, 70.0, 90.0, 95.0, 90.0),
                "reindex_matrix": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
            },
            {
                "character": "oP",
                "unit_cell": (50.1, 60.1, 70.1, 90.0, 90.0, 90.0),
                "reindex_matrix": (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0),
            },
        ]
        matched, matrix = find_lattice(target, candidates)
        self.assertEqual(matched.spacegroup, 19)
        self.assertEqual(matched.a, 50.1)
        self.assertEqual(matrix, (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0))

    def test_find_lattice_returns_none_when_missing(self):
        target = Lattice(19, 50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
        self.assertEqual(find_lattice(target, []), (None, None))


if __name__ == "__main__":
    unittest.main()
