"""Heuristic 0.0-1.0 data quality scoring for rugnux MX datasets."""

from __future__ import annotations

import math

from .models import RugnuxReport

# Pathology flags that reflect actual data-quality degradation vs metadata advisories.
_DATA_DEGRADING_FLAGS = {
    "LOW_COMPLETENESS",
    "TWINNING",
    "MULTIPLE_LATTICES",
    "ICE_RINGS",
    "POWDER_RINGS",
    "OVERLOADS",
    "OVERLAPS",
    "SWEEP_GAPS",
    "SCALING_NOT_CONVERGED",
    "GEOMETRY_NOT_CONVERGED",
    "ANISOTROPY",
    "PSEUDO_TRANSLATION",
    "LATTICE_TRANSLATION",
    "SUPERCELL_POSSIBLE",
    "HARMONIC_CONTAMINATION",
    "LATTICE_EXPLAINS_FEW_SPOTS",
    "SWEEP_LEFT_OUT",
    "RESOLUTION_FIT",
}

_ADVISORY_FLAGS = {
    "BEAM_CENTRE_OFF",
    "INDEXING_AMBIGUITY",
    "SYMMETRY_AMBIGUITY",
    "CENTERING_UNTESTED",
    "FLIGHT_PATH",
    "GONIO_SCALE",
    "SPINDLE_CAP",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_score(report: RugnuxReport) -> float:
    """Compute a 0.0-1.0 quality score from rugnux merging statistics and verdict.

    Returns 0.0 when the run failed or did not perform merging.
    Missing/unmeasured metrics (such as `ISA= undetermined`) are excluded and the
    remaining component weights are re-normalized.
    """
    verdict = report.verdict
    if verdict == "FAILED" or not report.is_merged:
        return 0.0

    components: list[tuple[float, float]] = []  # (weight, subscore)

    # 1. Resolution (prefer FITTED_RESOLUTION; fallback to written high-res cutoff)
    res = report.get_float("FITTED_RESOLUTION")
    if res is None:
        res_range = report.get_floats("INCLUDE_RESOLUTION_RANGE")
        if len(res_range) >= 2:
            res = res_range[1]
    if res is not None and res > 0.0:
        res_score = 1.0 / (1.0 + math.exp((res - 2.9) / 0.55))
        components.append((0.25, _clamp(res_score)))

    # 2. Completeness (0..100%)
    completeness = report.get_float("COMPLETENESS")
    if completeness is not None:
        components.append((0.20, _clamp(completeness / 100.0)))

    # 3. Mean I/sigma(I)
    i_over_sigma = report.get_float("I_OVER_SIGMA")
    if i_over_sigma is not None:
        if i_over_sigma <= 0.0:
            ios_score = 0.0
        else:
            ios_score = i_over_sigma / (i_over_sigma + 2.5)
        components.append((0.20, _clamp(ios_score)))

    # 4. Overall CC1/2 (fraction 0..1)
    cc_half = report.get_float("CC_HALF")
    if cc_half is not None:
        components.append((0.15, _clamp(cc_half)))

    # 5. Overall R_meas (fraction, e.g. 0.0383)
    r_meas = report.get_float("R_MEAS")
    if r_meas is not None:
        r_score = math.exp(-2.5 * max(0.0, r_meas))
        components.append((0.10, _clamp(r_score)))

    # 6. ISa (optional; omitted when undetermined)
    isa = report.get_float("ISA")
    if isa is not None and isa > 0.0:
        isa_score = isa / (isa + 5.0)
        components.append((0.10, _clamp(isa_score)))

    if not components:
        return 0.0

    total_weight = sum(w for w, _ in components)
    score = sum(w * s for w, s in components) / total_weight

    # Apply mild penalties for pathology flags
    penalty = 0.0
    for flag in report.pathology_flags:
        if flag in _DATA_DEGRADING_FLAGS:
            penalty += 0.02
        elif flag in _ADVISORY_FLAGS:
            penalty += 0.01
        else:
            penalty += 0.015
    score -= min(penalty, 0.15)

    if verdict == "UNUSABLE":
        score = min(score, 0.35)

    return _clamp(score)
