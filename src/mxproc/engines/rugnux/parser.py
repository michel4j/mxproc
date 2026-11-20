"""Parsers and CLI helpers for rugnux `<prefix>_report.txt` and `<prefix>_plot.txt`."""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Sequence

from .models import ExtraLattice, IceRingRow, PlotTable, RugnuxReport, ShellRow, SweepEntry


_KV_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=\s*(.*)$")
_BANNER_STARS_RE = re.compile(r"^#\s*\*{10,}\s*$")
_SUMMARY_FACT_RE = re.compile(r"^#\s{2}([A-Z][A-Za-z0-9 /_-]+?)\s{2,}(.+)$")
_DASH_LINE_RE = re.compile(r"^#\s*-{3,}[\s-]+$")
_SWEEP_KEY_RE = re.compile(r"^SWEEP_(\d+)_(.+)$")
_EXTRA_LATTICE_KEY_RE = re.compile(r"^EXTRA_LATTICE_(\d+)_(.+)$")
_ICE_RING_KEY_RE = re.compile(r"^ICE_INTENSITY_RING_(\d+)$")
_TWIN_LAW_KEY_RE = re.compile(r"^TWIN_LAW_(\d+)$")

_INT_PLOT_COLS = {"image", "spots", "cc_n", "merged"}


def resolve_rugnux_binary() -> str:
    """Resolve the rugnux executable path from env, PATH, or default install location."""
    if env_bin := os.environ.get("RUGNUX_BIN"):
        return env_bin
    if which_bin := shutil.which("rugnux"):
        return which_bin
    return "rugnux"


def build_rugnux_command(
    inputs: Sequence[Path | str],
    *,
    prefix: str = "rugnux",
    anomalous: bool = False,
    spacegroup: int | str | None = None,
    resolution: float | None = None,
    beam_center: Sequence[float] | None = None,
    frames: Sequence[tuple[int, int]] | None = None,
    no_merge: bool = False,
    binary: str | None = None,
) -> str:
    """Build a rugnux CLI command string for the given inputs and options."""
    exe = binary or resolve_rugnux_binary()
    parts = [shlex.quote(exe), "-o", shlex.quote(prefix)]

    if anomalous:
        parts.append("-A")
    if spacegroup:
        parts.extend(["-S", shlex.quote(str(spacegroup))])
    if resolution is not None:
        parts.extend(["--scaling-high-resolution", str(resolution)])
    if beam_center and len(beam_center) >= 2:
        parts.extend(["--beam-x", str(beam_center[0]), "--beam-y", str(beam_center[1])])
    if frames:
        start = max(0, int(frames[0][0]) - 1)
        end = int(frames[-1][1])
        parts.extend(["-s", str(start), "-e", str(end)])
    if no_merge:
        parts.append("--no-merge")

    parts.extend(shlex.quote(str(item)) for item in inputs)
    return " ".join(parts)


def _parse_optional_float(token: str) -> float | None:
    cleaned = token.strip().rstrip("%")
    if cleaned in {"", "-", "nan", "NaN", "NAN", "inf", "-inf", "None", "undetermined"}:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return val


def _parse_optional_int(token: str) -> int | None:
    cleaned = token.strip()
    if cleaned in {"", "-", "nan", "NaN", "NAN", "None"}:
        return None
    try:
        return int(cleaned)
    except ValueError:
        val = _parse_optional_float(cleaned)
        return int(round(val)) if val is not None else None


def _parse_shell_row(headers: list[str], tokens: list[str]) -> ShellRow | None:
    if len(tokens) < 2:
        return None
    is_overall = tokens[0].lower() == "overall"
    if not is_overall and _parse_optional_float(tokens[0]) is None:
        return None

    col_map: dict[str, str] = {}
    for idx, col_name in enumerate(headers):
        if idx < len(tokens):
            col_map[col_name] = tokens[idx]

    label = "Overall" if is_overall else tokens[0]
    return ShellRow(
        label=label,
        d_min=None if is_overall else _parse_optional_float(col_map.get("d_min", tokens[0])),
        n_obs=_parse_optional_int(col_map.get("N_obs", "-")),
        n_uniq=_parse_optional_int(col_map.get("N_uniq", "-")),
        n_possib=_parse_optional_int(col_map.get("N_possib", "-")),
        completeness=_parse_optional_float(col_map.get("Compl", "-")),
        multiplicity=_parse_optional_float(col_map.get("Mult", "-")),
        i_over_sigma=_parse_optional_float(col_map.get("<I/sig>", "-")),
        r_merge=_parse_optional_float(col_map.get("R_merge", "-")),
        r_meas=_parse_optional_float(col_map.get("R_meas", "-")),
        r_pim=_parse_optional_float(col_map.get("R_pim", "-")),
        cc_half=_parse_optional_float(col_map.get("CC1/2", "-")),
        cc_star=_parse_optional_float(col_map.get("CC*", "-")),
        cc_ref=_parse_optional_float(col_map.get("CCref", "-")),
        cc_anom=_parse_optional_float(col_map.get("CCanom", "-")),
        sig_ano=_parse_optional_float(col_map.get("SigAno", "-")),
        is_overall=is_overall,
    )


def parse_report_text(text: str, source_path: Path | None = None) -> RugnuxReport:
    """Parse the text contents of a rugnux `<prefix>_report.txt` file."""
    report = RugnuxReport(source_path=source_path)
    lines = text.splitlines()

    current_section = "HEADER"
    report.sections.setdefault(current_section, {})
    shell_context = "primary"

    sweeps_by_idx: dict[int, SweepEntry] = {}
    extra_lattices_by_idx: dict[int, ExtraLattice] = {}
    twin_laws_by_idx: list[tuple[int, str]] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if (
            _BANNER_STARS_RE.match(line)
            and i + 2 < n
            and lines[i + 1].startswith("#")
            and _BANNER_STARS_RE.match(lines[i + 2])
        ):
            sec_title = lines[i + 1].lstrip("#").strip()
            if sec_title and sec_title != "END OF REPORT":
                current_section = sec_title
                report.sections.setdefault(current_section, {})
                if current_section.startswith("3."):
                    shell_context = "primary"
            i += 3
            continue

        if not stripped:
            i += 1
            continue

        kv_match = _KV_RE.match(line)
        if kv_match:
            key, value = kv_match.group(1), kv_match.group(2).strip()
            report.kv[key] = value
            report.sections.setdefault(current_section, {})[key] = value

            if current_section.startswith("3."):
                if key.startswith("ANOM_REFRES_"):
                    shell_context = "anom_refres"
                elif key.startswith("REFRES_"):
                    shell_context = "refres"
                elif key in {
                    "ANOM_UNIQUE_REFLECTIONS",
                    "ANOM_TOTAL_OBSERVATIONS",
                    "ANOM_COMPLETENESS",
                }:
                    shell_context = "anom"

            if m_sweep := _SWEEP_KEY_RE.match(key):
                idx = int(m_sweep.group(1))
                entry = sweeps_by_idx.setdefault(idx, SweepEntry(index=idx))
                entry.fields[m_sweep.group(2)] = value

            if m_extra := _EXTRA_LATTICE_KEY_RE.match(key):
                idx = int(m_extra.group(1))
                entry_l = extra_lattices_by_idx.setdefault(idx, ExtraLattice(index=idx))
                entry_l.fields[m_extra.group(2)] = value

            if m_ice := _ICE_RING_KEY_RE.match(key):
                parts = value.split()
                if len(parts) >= 5:
                    d_sp = _parse_optional_float(parts[0])
                    n_ref = _parse_optional_int(parts[1])
                    mean_r = _parse_optional_float(parts[2])
                    z_sc = _parse_optional_float(parts[3])
                    med_r = _parse_optional_float(parts[4])
                    if None not in (d_sp, n_ref, mean_r, z_sc, med_r):
                        report.ice_intensity_rings.append(
                            IceRingRow(
                                index=int(m_ice.group(1)),
                                d_spacing=d_sp,  # type: ignore[arg-type]
                                n_refl=n_ref,  # type: ignore[arg-type]
                                mean_ratio=mean_r,  # type: ignore[arg-type]
                                z_score=z_sc,  # type: ignore[arg-type]
                                median_ratio=med_r,  # type: ignore[arg-type]
                            )
                        )

            if m_twin := _TWIN_LAW_KEY_RE.match(key):
                twin_laws_by_idx.append((int(m_twin.group(1)), value))

            i += 1
            continue

        if line.startswith("#"):
            if line.startswith("# WARNING:"):
                report.warnings.append(line[len("# WARNING:") :].strip())
                i += 1
                continue

            if line.startswith("# XDS.INP>"):
                report.xds_geometry_lines.append(line[len("# XDS.INP>") :].strip())
                i += 1
                continue

            comment_body = line[1:].strip()
            if comment_body.startswith("Wilson B-factor estimate:"):
                report.wilson_note = comment_body
                i += 1
                continue
            if comment_body.startswith("Radiation damage:"):
                report.radiation_note = comment_body
                i += 1
                continue

            if current_section == "SUMMARY":
                if m_fact := _SUMMARY_FACT_RE.match(line):
                    label = m_fact.group(1).strip()
                    val = m_fact.group(2).strip()
                    if label.lower() == "advice":
                        report.advice_lines.append(val)
                    else:
                        report.summary_lines[label] = val
                    i += 1
                    continue

            if current_section.startswith("3.") and comment_body.startswith("d_min"):
                headers = comment_body.split()
                if "N_obs" in headers and "Compl" in headers and i + 1 < n and _DASH_LINE_RE.match(lines[i + 1]):
                    i += 2
                    shells: list[ShellRow] = []
                    overall_row: ShellRow | None = None
                    saw_bottom_rule = False
                    while i < n:
                        row_line = lines[i]
                        if not row_line.startswith("#"):
                            break
                        if _DASH_LINE_RE.match(row_line):
                            saw_bottom_rule = True
                            i += 1
                            continue
                        row_body = row_line[1:].strip()
                        if not row_body:
                            i += 1
                            break
                        if row_body.startswith(("Wilson B-factor", "Radiation damage")):
                            break
                        parsed_row = _parse_shell_row(headers, row_body.split())
                        if parsed_row is None:
                            if saw_bottom_rule:
                                break
                            i += 1
                            continue
                        if parsed_row.is_overall:
                            overall_row = parsed_row
                            i += 1
                            break
                        shells.append(parsed_row)
                        i += 1

                    if shell_context == "primary" and not report.primary_shells:
                        report.primary_shells = shells
                        report.primary_overall = overall_row
                    elif shell_context == "anom" and not report.anom_shells:
                        report.anom_shells = shells
                        report.anom_overall = overall_row
                    elif shell_context == "refres" and not report.refres_shells:
                        report.refres_shells = shells
                        report.refres_overall = overall_row
                    continue

        i += 1

    report.sweeps = [sweeps_by_idx[k] for k in sorted(sweeps_by_idx)]
    report.extra_lattices = [extra_lattices_by_idx[k] for k in sorted(extra_lattices_by_idx)]
    report.twin_laws = [text_val for _, text_val in sorted(twin_laws_by_idx)]
    return report


def parse_report(path: str | Path) -> RugnuxReport:
    """Read and parse a rugnux `<prefix>_report.txt` file."""
    p = Path(path)
    return parse_report_text(p.read_text(encoding="utf-8", errors="replace"), source_path=p)


def parse_plot_text(text: str, source_path: Path | None = None) -> PlotTable:
    """Parse the text contents of a rugnux `<prefix>_plot.txt` file."""
    table = PlotTable(source_path=source_path)
    columns: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if not columns:
                header_tokens = stripped.lstrip("#").strip().split()
                if header_tokens:
                    columns = header_tokens
                    table.columns = columns
                    table.rows = {col: [] for col in columns}
            continue

        if not columns:
            columns = [
                "image",
                "angle_deg",
                "bkg",
                "resolution_A",
                "spots",
                "scale_G",
                "sigma_M_deg",
                "cc_to_merge",
                "cc_n",
                "merged",
                "disposition",
            ]
            table.columns = columns
            table.rows = {col: [] for col in columns}

        tokens = stripped.split()
        for idx, col in enumerate(columns):
            tok = tokens[idx] if idx < len(tokens) else "nan"
            if col in _INT_PLOT_COLS:
                table.rows[col].append(_parse_optional_int(tok))
            else:
                table.rows[col].append(_parse_optional_float(tok))

    return table


def parse_plot(path: str | Path) -> PlotTable:
    """Read and parse a rugnux `<prefix>_plot.txt` file."""
    p = Path(path)
    return parse_plot_text(p.read_text(encoding="utf-8", errors="replace"), source_path=p)
