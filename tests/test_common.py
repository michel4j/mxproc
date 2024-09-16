import unittest
from mxproc import common

KEY_DICT = {
    'beam_axis': (-0.001557, -0.000442, 0.967838),
    'cell_a_axis': (-27.239513, 50.644436, 11.521472),
    'cell_b_axis': (20.310566, -1.194186, 55.59893),
    'cell_c_axis': (123.726677, 76.550972, -43.787224),
    'delta_angle': 0.2,
    'detector_normal': (0.0, 0.0, 1.0),
    'detector_origin': (1201.0, 1299.0),
    'detector_size': (2463, 2527),
    'detector_x_axis': (1.0, 0.0, 0.0),
    'detector_y_axis': (0.0, 1.0, 0.0),
    'distance': 298.299988,
    'first_frame': 1,
    'point_groups': ['P1', 'P2', 'C2', 'P222', 'C222', 'P4', 'P422'],
    'quality': {
        'angle_error': 0.26,
        'half_percent': 0.0,
        'index_deviation': 0.003,
        'max_deviation': 0.05,
        'misfit_percent': 13.5,
        'mosaicity': 0.2,
        'pixel_error': 1.56,
        'subtree_ratio': 0.0,
        'expected_error': {
            'pixel':  1.65,
            'angle': 0.257,
        }
    },
    'rotation_axis': (1.0, 6e-05, -9.2e-05),
    'segments': 1,
    'spots': {
        'best_range': (53, 97),
        'indexed': 1728,
        'misfits': 270,
        'overlap': 0,
        'total': 1998
    },
    'start_angle': 0.0,
    'subtrees': [1909, 3, 2, 2, 2, 2, 2, 2, 1, 1],
    'wavelength': 1.03323
}


class ResultTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.result = common.Result(details=KEY_DICT)

    def test_absent_default(self):
        assert self.result.get('__not_present__', -999) == -999, "Failed to get default value for missing key"

    def test_single_level(self):
        assert self.result.get('wavelength') == 1.03323, "Failed to get existing value"

    def test_deep_level(self):
        assert self.result.get('quality.expected_error.pixel') == 1.65, "Failed to get 3-level value"

    def test_partial_resolution(self):
        assert self.result.get('wavelength.__nothing__') is None, "Failed to return None if only first level exists"

    def test_present_with_default(self):
        assert self.result.get('quality.index_deviation', -1) == 0.003, "Failed to get existing value, with default provided"

    def test_missing_no_default(self):
        assert self.result.get('__not_present__') is None, "Failed to return None for missing value no default provided"



