"""Build typed ReportPayload instances from XDS engine step results."""

from __future__ import annotations

from typing import Any

from mxproc.reporting.models import (
    ExperimentReportData,
    IndexReportData,
    IntegrationReportData,
    QualitySummary,
    ReportPayload,
    ScalingReportData,
    StrategyReportData,
    SymmetryReportData,
    TwinningReportData,
)
from mxproc.xtal import Lattice


def build_report_payload(analysis: Any) -> ReportPayload:
    """
    Build a typed ReportPayload from an XDSAnalysis instance's step results.
    """
    experiments: list[ExperimentReportData] = []
    for expt in analysis.experiments:
        idx_res = analysis.get_step_result(expt, "index")
        strat_res = analysis.get_step_result(expt, "strategy")
        int_res = analysis.get_step_result(expt, "integrate")
        sym_res = analysis.get_step_result(expt, "symmetry")
        scale_res = analysis.get_step_result(expt, "scale")

        indexing = (
            IndexReportData(
                lattice=idx_res.get("lattice", Lattice()),
                symmetry_lattice=idx_res.get("symmetry_lattice", Lattice()),
                lattices=idx_res.get("lattices", []),
                point_groups=idx_res.get("point_groups", ()),
                pixel_error=idx_res.get("quality.pixel_error", 0.0),
                angle_error=idx_res.get("quality.angle_error", -1.0),
            )
            if idx_res
            else None
        )

        strategy = (
            StrategyReportData(
                score=strat_res.get("quality.score", 0.0),
                mosaicity=strat_res.get("quality.mosaicity", 0.0),
                resolution=strat_res.get("quality.resolution", 0.0),
                warnings=strat_res.get("quality.warnings", ()),
                strategy=strat_res.get("strategy", {}),
                statistics=strat_res.get("statistics", []),
            )
            if strat_res
            else None
        )

        integration = (
            IntegrationReportData(
                lattices=int_res.get("lattices", []),
                errors=int_res.get("quality.errors", []),
            )
            if int_res
            else None
        )

        symmetry = (
            SymmetryReportData(
                lattice=sym_res.get("lattice", Lattice()),
                candidates=sym_res.get("symmetry.candidates", []),
                summary=QualitySummary.from_result(sym_res),
                frames=sym_res.get("quality.frames", []),
                differences=sym_res.get("quality.differences", []),
            )
            if sym_res
            else None
        )

        scaling = (
            ScalingReportData(
                lattice=scale_res.get("lattice", Lattice()),
                summary=QualitySummary.from_result(scale_res),
                statistics=scale_res.get("quality.statistics", []),
                twinning=TwinningReportData(
                    plots=scale_res.get("quality.twinning.plots", []),
                    l_zscore=scale_res.get("quality.twinning.l_zscore"),
                    l_statistic=scale_res.get("quality.twinning.l_statistic", ()),
                    l_test=scale_res.get("quality.twinning.l_test", []),
                    laws=scale_res.get("quality.twinning.laws", []),
                ),
            )
            if scale_res
            else None
        )

        work_dir = analysis.options.working_directories.get(expt.identifier, analysis.options.directory)
        experiments.append(
            ExperimentReportData(
                name=expt.name,
                wavelength=expt.wavelength,
                work_dir=work_dir,
                indexing=indexing,
                strategy=strategy,
                integration=integration,
                symmetry=symmetry,
                scaling=scaling,
            )
        )

    return ReportPayload(
        directory=analysis.options.directory,
        anomalous=analysis.options.anomalous,
        screen=analysis.options.screen,
        merge=analysis.options.merge,
        symmetry_method=analysis.settings.get("symmetry_method", ""),
        resolution_method=analysis.settings.get("resolution_method", ""),
        outputs=analysis.settings.get("reflection_files", {}),
        experiments=experiments,
    )
