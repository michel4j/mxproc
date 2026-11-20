from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from mxio import XYPair
from mxproc import Analysis, reporting
from mxproc.common import Result, StateType, short_path
from mxproc.engines.rugnux.models import PlotTable, RugnuxReport
from mxproc.engines.rugnux.parser import (
    build_rugnux_command,
    parse_plot,
    parse_report,
)
from mxproc.engines.rugnux.report import build_report
from mxproc.engines.rugnux.scoring import compute_score
from mxproc.log import logger
from mxproc.xtal import Lattice, lattice_point_groups, spacegroup_character


class RUGNUXAnalysis(Analysis):
    prefix = "rugnux"

    def __init__(self, args: argparse.Namespace, runner=None):
        super().__init__(args, runner=runner)
        self._reports: dict[str, tuple[RugnuxReport, PlotTable | None]] = {}

    def get_extras(self, args: argparse.Namespace) -> dict:
        extras: dict[str, Any] = {}
        if args.frames:
            extras["frames"] = args.frames
        if args.beam_center:
            extras["beam_center"] = XYPair(*args.beam_center)
        if args.resolution:
            extras["resolution_limit"] = args.resolution
        if args.spacegroup:
            extras["lattice"] = Lattice(spacegroup=args.spacegroup)
        return extras

    def initialize(self, **kwargs):
        logger.info("Preparing Working directories:")
        self.prepare_experiment_directories()
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            logger.info_value(f" - {experiment.name}", str(directory))
        self.settings["symmetry_method"] = "determined by Rugnux"
        self.settings["resolution_method"] = "CC1/2 fall-off by Rugnux"
        return {}

    def _run_dir_for(self, experiment) -> Path:
        if self.options.merge:
            return self.options.directory
        return self.options.working_directories.get(experiment.identifier, self.options.directory)

    def _build_command_args(self) -> dict[str, Any]:
        lattice = self.options.extras.get("lattice")
        beam_center = self.options.extras.get("beam_center")
        return {
            "prefix": self.prefix,
            "anomalous": self.options.anomalous,
            "spacegroup": lattice.spacegroup if lattice and lattice.spacegroup > 1 else None,
            "resolution": self.options.extras.get("resolution_limit"),
            "beam_center": (beam_center.x, beam_center.y) if beam_center else None,
            "frames": self.options.extras.get("frames"),
            "no_merge": self.options.screen,
        }

    def _load_outputs(self, work_dir: Path) -> tuple[RugnuxReport, PlotTable | None]:
        rep = parse_report(work_dir / f"{self.prefix}_report.txt")
        plot_path = work_dir / f"{self.prefix}_plot.txt"
        plot = parse_plot(plot_path) if plot_path.is_file() else None
        return rep, plot

    def _ensure_rugnux(self) -> dict[str, tuple[RugnuxReport, PlotTable | None]]:
        if self._reports and all(expt.identifier in self._reports for expt in self.experiments):
            return self._reports

        cmd_kwargs = self._build_command_args()
        report_name = f"{self.prefix}_report.txt"

        if self.options.merge and len(self.experiments) > 1:
            work_dir = self.options.directory
            report_path = work_dir / report_name
            if not report_path.exists():
                inputs = [expt.directory / expt.reference for expt in self.experiments]
                cmd = build_rugnux_command(inputs, **cmd_kwargs)
                names = ", ".join(expt.name for expt in self.experiments)
                self.runner.run(
                    cmd,
                    work_dir,
                    desc=f"Running Rugnux on {names}",
                    check_files=[report_name],
                )
            parsed = self._load_outputs(work_dir)
            for expt in self.experiments:
                self._reports[expt.identifier] = parsed
            return self._reports

        for expt in self.experiments:
            if expt.identifier in self._reports:
                continue
            work_dir = self._run_dir_for(expt)
            report_path = work_dir / report_name
            if not report_path.exists():
                cmd = build_rugnux_command([expt.directory / expt.reference], **cmd_kwargs)
                self.runner.run(
                    cmd,
                    work_dir,
                    desc=f"{expt.name}: Running Rugnux",
                    check_files=[report_name],
                )
            self._reports[expt.identifier] = self._load_outputs(work_dir)

        return self._reports

    @staticmethod
    def _state_for(rep: RugnuxReport) -> StateType:
        if rep.verdict == "FAILED":
            return StateType.FAILURE
        if rep.warnings or rep.verdict in {"WARNINGS", "UNUSABLE"}:
            return StateType.WARNING
        return StateType.SUCCESS

    @staticmethod
    def _lattice_for(rep: RugnuxReport) -> Lattice:
        sg_number = rep.get_int("SPACE_GROUP_NUMBER", 1) or 1
        cell = rep.get_floats("UNIT_CELL_CONSTANTS")
        if len(cell) >= 6:
            return Lattice(sg_number, *cell[:6])
        return Lattice(sg_number)

    @staticmethod
    def _resolution_for(rep: RugnuxReport) -> float:
        fitted = rep.get_float("FITTED_RESOLUTION")
        if fitted is not None:
            return fitted
        res_range = rep.get_floats("INCLUDE_RESOLUTION_RANGE")
        if len(res_range) >= 2:
            return res_range[1]
        return 20.0

    @staticmethod
    def _candidates_for(rep: RugnuxReport, lattice: Lattice) -> list[dict[str, Any]]:
        sg_number = lattice.spacegroup
        sg_name = rep.get_str("SPACE_GROUP_NAME") or lattice.name
        candidates: list[dict[str, Any]] = [{"name": sg_name, "number": sg_number, "probability": 1.0}]
        for key in ("SPACE_GROUP_ALTERNATIVES", "SPACE_GROUP_ACHIRAL_ALTERNATIVES"):
            raw = rep.get_str(key)
            if not raw:
                continue
            for alt in raw.split(","):
                alt_name = alt.strip()
                if alt_name and alt_name != sg_name:
                    candidates.append({"name": alt_name, "number": sg_number, "probability": 0.5})
        return candidates

    @staticmethod
    def _shell_dicts(rep: RugnuxReport) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for shell in rep.primary_shells:
            rows.append({
                "shell": shell.d_min or 0.0,
                "observed": shell.n_obs or 0,
                "unique": shell.n_uniq or 0,
                "completeness": shell.completeness or 0.0,
                "r_obs": shell.r_merge or 0.0,
                "r_meas": shell.r_meas or 0.0,
                "cc_half": shell.cc_half or 0.0,
                "i_sigma": shell.i_over_sigma or 0.0,
                "sig_ano": shell.sig_ano or 0.0,
                "cor_ano": shell.cc_anom or 0.0,
            })
        return rows

    def _summary_dict(self, rep: RugnuxReport) -> dict[str, Any]:
        cc_half = rep.get_float("CC_HALF")
        r_meas = rep.get_float("R_MEAS")
        if r_meas is None:
            r_meas = rep.get_float("R_SPLIT")

        return {
            "score": self.score(rep),
            "resolution": self._resolution_for(rep),
            "observed": rep.get_int("TOTAL_OBSERVATIONS", 0) or 0,
            "unique": rep.get_int("UNIQUE_REFLECTIONS", 0) or 0,
            "completeness": rep.get_float("COMPLETENESS", 0.0) or 0.0,
            "anom_completeness": rep.get_float("ANOM_COMPLETENESS", 0.0) or 0.0,
            "mosaicity": rep.get_float("MOSAICITY_DEG"),
            "i_sigma": rep.get_float("I_OVER_SIGMA", -99.0),
            "r_meas": r_meas * 100.0 if r_meas is not None else -99.0,
            "cc_half": cc_half * 100.0 if cc_half is not None else -99.0,
            "i_sigma_a": rep.get_float("ISA"),
        }

    def find_spots(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={"images_processed": rep.get_int("IMAGES_PROCESSED", 0) or 0},
            )
        return results

    def index(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            lattice = self._lattice_for(rep)
            character = spacegroup_character(lattice.spacegroup)
            lattices = [
                {
                    "index": 1,
                    "character": character,
                    "quality": 0.0,
                    "unit_cell": (lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma),
                }
            ]
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={
                    "lattice": lattice,
                    "symmetry_lattice": lattice,
                    "lattices": lattices,
                    "point_groups": lattice_point_groups(character),
                    "quality": {"pixel_error": 0.0, "angle_error": -1.0},
                },
            )
        return results

    def strategy(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            summary = self._summary_dict(rep)
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={
                    "quality": {
                        "score": summary["score"],
                        "mosaicity": rep.get_float("MOSAICITY_DEG", 0.0) or 0.0,
                        "resolution": summary["resolution"],
                        "warnings": tuple(rep.warnings),
                    },
                    "strategy": {"resolution": summary["resolution"]},
                    "statistics": [],
                },
            )
        return results

    def integrate(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            lattice = self._lattice_for(rep)
            character = spacegroup_character(lattice.spacegroup)
            lattices = [
                {
                    "index": 1,
                    "character": character,
                    "quality": 0.0,
                    "unit_cell": (lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma),
                }
            ]
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={
                    "lattices": lattices,
                    "quality": {"errors": []},
                },
            )
        return results

    def symmetry(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            lattice = self._lattice_for(rep)
            summary = self._summary_dict(rep)
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={
                    "lattice": lattice,
                    "symmetry": {"candidates": self._candidates_for(rep, lattice)},
                    "quality": {
                        "summary": summary,
                        "frames": [],
                        "differences": [],
                    },
                },
            )
        return results

    def scale(self, **kwargs) -> Dict[str, Result]:
        reports = self._ensure_rugnux()
        results: Dict[str, Result] = {}
        for expt in self.experiments:
            rep, _ = reports[expt.identifier]
            lattice = self._lattice_for(rep)
            summary = self._summary_dict(rep)
            self.show_quality(expt.name, lattice, summary)
            results[expt.identifier] = Result(
                state=self._state_for(rep),
                messages=tuple(rep.warnings),
                details={
                    "lattice": lattice,
                    "quality": {
                        "summary": summary,
                        "statistics": self._shell_dicts(rep),
                        "twinning": {},
                    },
                },
            )
        return results

    def export(self, **kwargs) -> Dict[str, Result]:
        self._ensure_rugnux()
        results: Dict[str, Result] = {}
        all_outputs: dict[str, str] = {}
        candidates = (
            ("mtz", f"{self.prefix}.mtz"),
            ("unmerged_mtz", f"{self.prefix}_unmerged.mtz"),
            ("cif", f"{self.prefix}.cif"),
            ("hkl", f"{self.prefix}.hkl"),
            ("report", f"{self.prefix}_report.txt"),
            ("plot", f"{self.prefix}_plot.txt"),
        )
        for expt in self.experiments:
            run_dir = self._run_dir_for(expt)
            expt_outputs: dict[str, str] = {}
            for key, filename in candidates:
                path = run_dir / filename
                if path.exists():
                    rel = short_path(path, self.options.directory)
                    expt_outputs[key] = str(rel)
                    all_outputs[key] = str(rel)
            results[expt.identifier] = Result(state=StateType.SUCCESS, details=expt_outputs)

        self.settings["reflection_files"] = all_outputs
        return results

    @staticmethod
    def show_quality(name: str, lattice: Lattice, summary: dict[str, Any]):
        logger.info_value(f"{name} Space group", f"{lattice.name} ({lattice.spacegroup})")
        logger.info_value(f"{name} Unit cell", lattice.cell_text())
        logger.info_value(f"{name} Resolution", f"{summary['resolution']:0.2f} Å")
        logger.info_value(f"{name} Completeness", f"{summary['completeness']:0.1f} %")
        logger.info_value(f"{name} R-meas", f"{summary['r_meas']:0.1f} %")
        logger.info_value(f"{name} I/Sigma(I)", f"{summary['i_sigma']:0.1f}")
        logger.info_value(f"{name} CC1/2", f"{summary['cc_half']:0.1f} %")
        if summary.get("i_sigma_a") is not None:
            logger.info_value(f"{name} ISa", f"{summary['i_sigma_a']:0.1f}")

    @staticmethod
    def score(info: RugnuxReport) -> float:
        """Calculate and return a 0.0-1.0 quality score for a parsed RugnuxReport."""
        return compute_score(info)

    def report(self, **kwargs):
        reports = self._ensure_rugnux()
        outputs = self.settings.get("reflection_files", {})

        if self.options.merge or len(self.experiments) == 1:
            expt = self.experiments[0]
            rep, plot = reports[expt.identifier]
            report_dict = build_report(rep, plot, directory=self.options.directory)
            reporting.save_report(report_dict, self.options.directory, outputs=outputs)
            html_path = self.options.directory / "report.html"
            logger.info(f"- HTML report: {short_path(html_path, self.options.directory.parent)}")
            return

        for expt in self.experiments:
            work_dir = self._run_dir_for(expt)
            rep, plot = reports[expt.identifier]
            report_dict = build_report(rep, plot, directory=work_dir)
            reporting.save_report(report_dict, work_dir, outputs=outputs)
            html_path = work_dir / "report.html"
            logger.info(f"- HTML report: {short_path(html_path, self.options.directory.parent)}")
