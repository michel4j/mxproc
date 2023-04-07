import unittest
from mxproc import common

KEY_DICT = {
    'angle_error': 0.26,
    'expected_angle_error': 0.257,
    'expected_pixel_error': 1.65,
    'half_percent': 0.0,
    'index_deviation': 0.003,
    'max_deviation': 0.05,
    'misfit_percent': 13.5,
    'mosaicity': 0.2,
    'pixel_error': 1.56,
    'subtree_ratio': 0.0
}


class ResultTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.result = common.Result(details=KEY_DICT)

    def get_absent_default(self):
        assert self.result.get('__not_present__', -999) == -999, "Failed to get default value for missing key"

    def get_get_present(self):
        assert self.result.get('angle_error') == 0.26, "Failed to get existing value"

    def get_present_with_default(self):
        assert self.result.get('index_deviation', -1) == 0.003, "Failed to get existing value, with default provided"

    def get_missing_no_default(self):
        assert self.result.get('__not_present__') is None, "Failed to return None for missing value no default provided"


if __name__ == '__main__':
    unittest.main()