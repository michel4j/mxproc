import json
import tempfile
import unittest
from pathlib import Path

from mxproc.command import CommandFailed
from mxproc.reporting import (
    ExperimentReportData,
    IndexReportData,
    IntegrationReportData,
    QualitySummary,
    ReportPayload,
    ScalingReportData,
    StrategyReportData,
    SymmetryReportData,
    TwinningReportData,
    generate_reports,
)
from mxproc.xtal import Lattice


def _sample_lattice() -> Lattice:
    return Lattice(19, 50.0, 60.0, 70.0, 90.0, 90.0, 90.0)


def _sample_lattices() -> list[dict]:
    return [
        {
            "index": 1,
            "character": "oP",
            "quality": 0.1,
            "unit_cell": (50.0, 60.0, 70.0, 90.0, 90.0, 90.0),
        }
    ]


def _sample_strategy() -> StrategyReportData:
    return StrategyReportData(
        score=0.85,
        mosaicity=0.15,
        resolution=1.8,
        warnings=("Low multiplicity",),
        strategy={
            "resolution": 1.8,
            "completeness": 99.0,
            "multiplicity": 3.5,
            "max_resolution": 1.6,
            "attenuation": 10.0,
            "start_angle": 0.0,
            "max_delta": 0.5,
            "total_angle": 90.0,
            "exposure_rate": 2.0,
            "exposure_rate_worst": 1.0,
            "total_exposure": 45.0,
            "total_exposure_worst": 90.0,
        },
        statistics=[
            {"start_angle": 0.0, "end_angle": 90.0, "completeness": 98.5},
            {"start_angle": 10.0, "end_angle": 100.0, "completeness": 99.0},
        ],
    )


def _sample_experiment(work_dir: Path, name: str = "expt1", screen: bool = False) -> ExperimentReportData:
    lat = _sample_lattice()
    if screen:
        return ExperimentReportData(
            name=name,
            wavelength=1.0,
            work_dir=work_dir,
            indexing=IndexReportData(
                lattice=lat,
                symmetry_lattice=lat,
                lattices=_sample_lattices(),
                point_groups=("222",),
                pixel_error=0.5,
                angle_error=0.1,
            ),
            strategy=_sample_strategy(),
        )

    summary = QualitySummary(
        score=0.91,
        resolution=1.8,
        observed=10000,
        unique=2500,
        completeness=99.2,
        anom_completeness=95.0,
        mosaicity=0.12,
        i_sigma=15.0,
        r_meas=5.2,
        cc_half=99.8,
        i_sigma_a=25.0,
    )
    return ExperimentReportData(
        name=name,
        wavelength=1.0,
        work_dir=work_dir,
        integration=IntegrationReportData(
            lattices=_sample_lattices(),
            errors=[
                {"resol_range": (20.0, 5.0), "r_obs": 2.0, "r_exp": 1.8, "chi_sq": 1.05, "i_sigma": 20.0},
                {"resol_range": (5.0, 2.0), "r_obs": 4.0, "r_exp": 3.8, "chi_sq": 1.02, "i_sigma": 5.0},
                {"resol_range": (20.0, 2.0), "r_obs": 3.0, "r_exp": 2.8, "chi_sq": 1.03, "i_sigma": 12.0},
            ],
        ),
        symmetry=SymmetryReportData(
            lattice=lat,
            candidates=[{"name": "P 21 21 21", "number": 19, "probability": 0.99}],
            summary=summary,
            frames=[
                {"frame": 1, "iobs": 100.0, "corr": 0.99, "r_meas": 4.0, "i_sigma": 15.0, "refs": 200, "unique": 150},
                {"frame": 2, "iobs": 98.0, "corr": 0.98, "r_meas": 4.2, "i_sigma": 14.0, "refs": 195, "unique": 148},
            ],
            differences=[
                {"frame_diff": 1, "rd": 0.02, "rd_friedel": 0.015, "rd_non_friedel": 0.025},
                {"frame_diff": 2, "rd": 0.03, "rd_friedel": 0.02, "rd_non_friedel": 0.035},
            ],
        ),
        scaling=ScalingReportData(
            lattice=lat,
            summary=summary,
            statistics=[
                {
                    "shell": 5.0,
                    "observed": 2000,
                    "unique": 500,
                    "possible": 500,
                    "completeness": 100.0,
                    "r_obs": 2.0,
                    "r_exp": 1.9,
                    "r_meas": 2.5,
                    "i_sigma": 30.0,
                    "cc_half": 99.9,
                    "anom_corr": 10.0,
                    "sig_ano": 1.1,
                    "cor_ano": 10.0,
                },
                {
                    "shell": 1.8,
                    "observed": 2000,
                    "unique": 500,
                    "possible": 505,
                    "completeness": 99.0,
                    "r_obs": 8.0,
                    "r_exp": 7.5,
                    "r_meas": 9.0,
                    "i_sigma": 4.0,
                    "cc_half": 95.0,
                    "anom_corr": 5.0,
                    "sig_ano": 0.9,
                    "cor_ano": 5.0,
                },
            ],
            twinning=TwinningReportData(
                plots=[{"inv_res_sq": 0.04, "mean_i": 100.0, "expected_i": 95.0, "mean_i_binned": 98.0}],
                l_zscore=1.2,
                l_statistic=(0.48, 0.50, 0.375),
                l_test=[{"abs_l": 0.1, "observed": 0.1, "twinned": 0.05, "untwinned": 0.1}],
                laws=[],
            ),
        ),
    )


class ReportingTests(unittest.TestCase):
    def test_generate_screening_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = ReportPayload(
                directory=root,
                screen=True,
                experiments=[_sample_experiment(root, screen=True)],
            )
            paths = generate_reports(payload)
            self.assertEqual(paths, [root / "report.html"])
            self.assertTrue((root / "report.json").exists())
            self.assertTrue((root / "report.txt").exists())
            self.assertTrue((root / "report.html").exists())
            data = json.loads((root / "report.json").read_text())
            self.assertEqual(data["kind"], "MX Screening")
            self.assertEqual(data["score"], 0.85)

    def test_generate_merging_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = ReportPayload(
                directory=root,
                merge=True,
                symmetry_method="automatically assigned",
                resolution_method="CC1/2",
                outputs={"mtz": "final.mtz"},
                experiments=[_sample_experiment(root, "ds1"), _sample_experiment(root, "ds2")],
            )
            paths = generate_reports(payload)
            self.assertEqual(paths, [root / "report.html"])
            data = json.loads((root / "report.json").read_text())
            self.assertIn("Merging of ds1, ds2", data["title"])
            self.assertEqual(data["outputs"], {"mtz": "final.mtz"})

    def test_generate_single_reports_per_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir1 = root / "ds1"
            dir2 = root / "ds2"
            dir1.mkdir()
            dir2.mkdir()
            payload = ReportPayload(
                directory=root,
                merge=False,
                anomalous=True,
                symmetry_method="automatically assigned",
                resolution_method="CC1/2",
                experiments=[_sample_experiment(dir1, "ds1"), _sample_experiment(dir2, "ds2")],
            )
            paths = generate_reports(payload)
            self.assertEqual(paths, [dir1 / "report.html", dir2 / "report.html"])
            self.assertTrue((dir1 / "report.json").exists())
            self.assertTrue((dir2 / "report.txt").exists())

    def test_raises_when_no_successful_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = ReportPayload(directory=Path(tmp), screen=False, experiments=[])
            with self.assertRaises(CommandFailed):
                generate_reports(payload)


if __name__ == "__main__":
    unittest.main()
