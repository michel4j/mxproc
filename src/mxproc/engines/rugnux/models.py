"""Data models for parsed rugnux report and plot files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math


@dataclass
class ShellRow:
    """A single resolution shell (or Overall) row from Section 3 of *_report.txt."""

    label: str
    d_min: float | None = None
    n_obs: int | None = None
    n_uniq: int | None = None
    n_possib: int | None = None
    completeness: float | None = None  # percentage (0..100)
    multiplicity: float | None = None
    i_over_sigma: float | None = None
    r_merge: float | None = None  # percentage
    r_meas: float | None = None  # percentage
    r_pim: float | None = None  # percentage
    cc_half: float | None = None  # percentage
    cc_star: float | None = None  # percentage
    cc_ref: float | None = None  # percentage
    cc_anom: float | None = None  # percentage
    sig_ano: float | None = None
    is_overall: bool = False


@dataclass
class SweepEntry:
    """One sweep entry from Section 1b. SWEEPS (`SWEEP_k_*`)."""

    index: int
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtraLattice:
    """One extra lattice entry from Section 4 (`EXTRA_LATTICE_n_*`)."""

    index: int
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class IceRingRow:
    """One ice-intensity ring row (`ICE_INTENSITY_RING_nn`)."""

    index: int
    d_spacing: float
    n_refl: int
    mean_ratio: float
    z_score: float
    median_ratio: float


@dataclass
class RugnuxReport:
    """Structured representation of a rugnux `<prefix>_report.txt` file."""

    source_path: Path | None = None
    kv: dict[str, str] = field(default_factory=dict)
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    summary_lines: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    advice_lines: list[str] = field(default_factory=list)
    primary_shells: list[ShellRow] = field(default_factory=list)
    primary_overall: ShellRow | None = None
    anom_shells: list[ShellRow] = field(default_factory=list)
    anom_overall: ShellRow | None = None
    refres_shells: list[ShellRow] = field(default_factory=list)
    refres_overall: ShellRow | None = None
    twin_laws: list[str] = field(default_factory=list)
    extra_lattices: list[ExtraLattice] = field(default_factory=list)
    ice_intensity_rings: list[IceRingRow] = field(default_factory=list)
    sweeps: list[SweepEntry] = field(default_factory=list)
    wilson_note: str | None = None
    radiation_note: str | None = None
    xds_geometry_lines: list[str] = field(default_factory=list)

    def get_str(self, key: str, default: str | None = None) -> str | None:
        val = self.kv.get(key)
        if val is None or val == "":
            return default
        return val

    def get_float(self, key: str, default: float | None = None) -> float | None:
        raw = self.kv.get(key)
        if raw is None:
            return default
        token = raw.strip().split()[0] if raw.strip() else ""
        if token.lower() in {"", "undetermined", "none", "nan", "inf", "-inf", "n/a", "-"}:
            return default
        try:
            value = float(token)
        except ValueError:
            return default
        if not math.isfinite(value):
            return default
        return value

    def get_int(self, key: str, default: int | None = None) -> int | None:
        raw = self.kv.get(key)
        if raw is None:
            return default
        token = raw.strip().split()[0] if raw.strip() else ""
        if token.lower() in {"", "undetermined", "none", "nan", "n/a", "-"}:
            return default
        try:
            return int(token)
        except ValueError:
            try:
                return int(float(token))
            except ValueError:
                return default

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        raw = self.kv.get(key)
        if raw is None:
            return default
        upper = raw.strip().upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False
        return default

    def get_floats(self, key: str) -> list[float]:
        raw = self.kv.get(key)
        if not raw:
            return []
        result: list[float] = []
        for token in raw.strip().split():
            try:
                val = float(token)
            except ValueError:
                continue
            if math.isfinite(val):
                result.append(val)
        return result

    @property
    def verdict(self) -> str:
        return (self.get_str("VERDICT") or "UNKNOWN").upper()

    @property
    def is_merged(self) -> bool:
        return (self.get_str("MERGE") or "").upper() == "PERFORMED"

    @property
    def is_anomalous_run(self) -> bool:
        return self.get_bool("FRIEDELS_LAW") is False

    def has_anomalous_signal(self) -> bool:
        """True if Friedel's law was disabled (-A) or rugnux detected anomalous signal."""
        if self.is_anomalous_run:
            return True
        anom_summary = self.summary_lines.get("Anomalous", "").lower()
        if "signal present" in anom_summary or "strong" in anom_summary:
            return True
        if "no usable signal" in anom_summary or "no signal" in anom_summary:
            return False
        if self.primary_shells and self.primary_shells[0].sig_ano is not None:
            if self.primary_shells[0].sig_ano >= 1.0:
                return True
        cc_anom = self.get_float("CC_ANOM")
        if cc_anom is not None and cc_anom >= 0.15:
            return True
        return False

    def dataset_name(self) -> str:
        """Derive a clean human-readable dataset name for the report title/header."""
        prefix = self.get_str("OUTPUT_PREFIX")
        if prefix and prefix.lower() not in {"output", "rugnux"}:
            return Path(prefix).name
        input_file = self.get_str("INPUT_FILE")
        if input_file:
            stem = Path(input_file).name
            for suffix in ("_master.h5", "_master.nxs", ".h5", ".nxs", ".cbf", ".img", ".mccd", ".smv"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            return stem or "dataset"
        if self.source_path is not None:
            name = self.source_path.name
            if name.endswith("_report.txt"):
                return name[: -len("_report.txt")]
            return self.source_path.stem
        return "dataset"

    @property
    def pathology_flags(self) -> list[str]:
        raw = self.get_str("PATHOLOGY_FLAGS")
        if not raw or raw.upper() == "NONE":
            return []
        return [tok for tok in raw.replace(",", " ").split() if tok and tok.upper() != "NONE"]

    @property
    def advice_codes(self) -> list[str]:
        raw = self.get_str("ADVICE")
        if not raw or raw.upper() == "NONE":
            return []
        return [tok for tok in raw.replace(",", " ").split() if tok and tok.upper() != "NONE"]


@dataclass
class PlotTable:
    """Structured representation of a rugnux `<prefix>_plot.txt` per-image table."""

    source_path: Path | None = None
    columns: list[str] = field(default_factory=list)
    rows: dict[str, list[float | int | None]] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        if not self.columns:
            return 0
        return len(self.rows.get(self.columns[0], []))

    def get_column(self, name: str) -> list[float | int | None]:
        return list(self.rows.get(name, []))

    @property
    def frame_numbers(self) -> list[int]:
        """1-based frame numbers for plotting (matches MxLIVE convention)."""
        images = self.rows.get("image")
        if images and all(v is not None for v in images):
            return [int(v) + 1 for v in images]  # type: ignore[arg-type]
        return list(range(1, self.n_frames + 1))

    @property
    def has_non_merged_disposition(self) -> bool:
        disp = self.rows.get("disposition", [])
        return any(v is not None and float(v) != 1.0 for v in disp)
