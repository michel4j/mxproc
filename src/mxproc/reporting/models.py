from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mxproc.common import Result
from mxproc.xtal import Lattice


@dataclass
class QualitySummary:
    score: float = 0.0
    resolution: float = 20.0
    observed: int = 0
    unique: int = 0
    completeness: float = 0.0
    anom_completeness: float = 0.0
    mosaicity: float | None = None
    i_sigma: float = -99.0
    r_meas: float = -99.0
    cc_half: float = -99.0
    i_sigma_a: float | None = None

    @classmethod
    def from_result(cls, result: Result) -> QualitySummary:
        return cls(
            score=result.get("quality.summary.score", 0.0),
            resolution=result.get("quality.summary.resolution", 20.0),
            observed=result.get("quality.summary.observed", 0),
            unique=result.get("quality.summary.unique", 0),
            completeness=result.get("quality.summary.completeness", 0.0),
            anom_completeness=result.get("quality.summary.anom_completeness", 0.0),
            mosaicity=result.get("quality.summary.mosaicity"),
            i_sigma=result.get("quality.summary.i_sigma", -99.0),
            r_meas=result.get("quality.summary.r_meas", -99.0),
            cc_half=result.get("quality.summary.cc_half", -99.0),
            i_sigma_a=result.get("quality.summary.i_sigma_a"),
        )


@dataclass
class IndexReportData:
    lattice: Lattice = field(default_factory=Lattice)
    symmetry_lattice: Lattice = field(default_factory=Lattice)
    lattices: list[dict] = field(default_factory=list)
    point_groups: Sequence[str] = ()
    pixel_error: float = 0.0
    angle_error: float = -1.0


@dataclass
class StrategyReportData:
    score: float = 0.0
    mosaicity: float = 0.0
    resolution: float = 0.0
    warnings: Sequence[str] = ()
    strategy: dict[str, Any] = field(default_factory=dict)
    statistics: list[dict] = field(default_factory=list)


@dataclass
class IntegrationReportData:
    lattices: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


@dataclass
class SymmetryReportData:
    lattice: Lattice = field(default_factory=Lattice)
    candidates: list[dict] = field(default_factory=list)
    summary: QualitySummary = field(default_factory=QualitySummary)
    frames: list[dict] = field(default_factory=list)
    differences: list[dict] = field(default_factory=list)


@dataclass
class TwinningReportData:
    plots: list[dict] = field(default_factory=list)
    l_zscore: float | None = None
    l_statistic: Sequence[float] = ()
    l_test: list[dict] = field(default_factory=list)
    laws: list[dict] = field(default_factory=list)


@dataclass
class ScalingReportData:
    lattice: Lattice = field(default_factory=Lattice)
    summary: QualitySummary = field(default_factory=QualitySummary)
    statistics: list[dict] = field(default_factory=list)
    twinning: TwinningReportData = field(default_factory=TwinningReportData)


@dataclass
class ExperimentReportData:
    name: str
    wavelength: float
    work_dir: Path
    indexing: IndexReportData | None = None
    strategy: StrategyReportData | None = None
    integration: IntegrationReportData | None = None
    symmetry: SymmetryReportData | None = None
    scaling: ScalingReportData | None = None


@dataclass
class ReportPayload:
    directory: Path
    anomalous: bool = False
    screen: bool = False
    merge: bool = True
    symmetry_method: str = ""
    resolution_method: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    experiments: list[ExperimentReportData] = field(default_factory=list)
