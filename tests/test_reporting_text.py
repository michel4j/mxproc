import os
import unittest
from unittest.mock import patch

from mxproc.reporting.text import make_tick_formatter, nice_ticks, plot, text_report


class NiceTicksTests(unittest.TestCase):
    def test_nice_ticks_expands_equal_bounds(self):
        ticks, limits, digits = nice_ticks(5.0, 5.0)
        self.assertLess(limits[0], 5.0)
        self.assertGreater(limits[1], 5.0)
        self.assertGreater(len(ticks), 1)
        self.assertGreaterEqual(digits, 0)

    def test_make_tick_formatter_default_and_non_default(self):
        ticks, _, digits = nice_ticks(0.0, 10.0, num_ticks=6)
        fmt_default = make_tick_formatter(ticks, digits, default=True)
        self.assertEqual(fmt_default(2.0, 4.0), "2")

        fmt_sparse = make_tick_formatter(ticks, digits, default=False)
        self.assertEqual(fmt_sparse(2.0, 4.0), "2")
        self.assertEqual(fmt_sparse(0.9, 1.0), "")


class PlotTests(unittest.TestCase):
    def test_plot_does_not_mutate_force_color_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            out = plot(
                {
                    "x": ["Shell", 5.0, 2.0],
                    "y1": [["Completeness", 100.0, 95.0]],
                    "x-scale": "inv-square",
                },
                plot_type="lineplot",
            )
            self.assertNotIn("FORCE_COLOR", os.environ)
            self.assertIn("Shell", out)
            self.assertIn("\x1b[", out)

    def test_text_report_renders_table_and_scatter(self):
        rendered = text_report([
            {
                "title": "Summary",
                "content": [
                    {
                        "title": "Metrics",
                        "kind": "table",
                        "header": "row",
                        "data": [["Name", "Score"], ["ds1", "0.9"]],
                    },
                    {
                        "kind": "scatterplot",
                        "style": "half-height",
                        "data": {"x": ["Frame", 1, 2], "y1": [["I", 10, 12]]},
                    },
                ],
            }
        ])
        self.assertIn("Summary", rendered)
        self.assertIn("ds1", rendered)
        self.assertIn("Frame", rendered)


if __name__ == "__main__":
    unittest.main()
