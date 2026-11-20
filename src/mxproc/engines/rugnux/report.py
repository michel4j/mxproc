"""Convert parsed rugnux outputs into MxLIVE/AutoProcess-compatible report.json dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import PlotTable, RugnuxReport, ShellRow
from .scoring import compute_score

def _fmt_float(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if value == 0.0:
        value = 0.0
    return f"{value:.{decimals}f}{suffix}"


def _fmt_pct_from_frac(value: float | None, decimals: int = 1, with_symbol: bool = True) -> str:
    if value is None:
        return "-"
    sym = " %" if with_symbol else ""
    return f"{value * 100.0:.{decimals}f}{sym}"


def _fmt_int(value: int | None) -> int | str:
    return value if value is not None else "-"


def _has_non_null(values: list[Any]) -> bool:
    return any(v is not None for v in values)


def _build_summary_stats_table(report: RugnuxReport, score: float) -> dict[str, Any]:
    dataset = report.dataset_name()
    rows: list[list[Any]] = [
        ["", dataset],
        ["Score\u00b9", f"{score:.2f}"],
        ["Verdict", report.verdict],
    ]

    wavelength = report.get_float("WAVELENGTH")
    if wavelength is not None:
        rows.append(["Wavelength (A)", f"{wavelength:.5f}".rstrip("0").rstrip(".")])

    sg_name = report.get_str("SPACE_GROUP_NAME")
    if sg_name:
        rows.append(["Space Group\u00b2", sg_name])

    unit_cell = report.get_str("UNIT_CELL_CONSTANTS")
    if unit_cell:
        rows.append(["Unit Cell (A)", unit_cell])

    fitted_res = report.get_float("FITTED_RESOLUTION")
    res_range = report.get_floats("INCLUDE_RESOLUTION_RANGE")
    if fitted_res is not None:
        rows.append(["Resolution\u2075 (A)", f"{fitted_res:.2f}"])
    elif len(res_range) >= 2:
        rows.append(["Resolution\u2075 (A)", f"{res_range[1]:.2f}"])

    total_obs = report.get_int("TOTAL_OBSERVATIONS")
    if total_obs is not None:
        rows.append(["All Reflections", total_obs])

    unique_refl = report.get_int("UNIQUE_REFLECTIONS")
    if unique_refl is not None:
        rows.append(["Unique Reflections", unique_refl])

    mult = report.get_float("MULTIPLICITY")
    if mult is not None:
        rows.append(["Multiplicity", f"{mult:.1f}"])

    compl = report.get_float("COMPLETENESS")
    if compl is not None:
        rows.append(["Completeness", f"{compl:.1f} %"])

    mosaicity = report.get_float("MOSAICITY_DEG")
    if mosaicity is not None:
        rows.append(["Mosaicity (deg)", f"{mosaicity:.3f}"])

    i_sig = report.get_float("I_OVER_SIGMA")
    if i_sig is not None:
        rows.append(["I/Sigma(I)", f"{i_sig:.1f}"])

    r_meas = report.get_float("R_MEAS")
    if r_meas is not None:
        rows.append(["R-meas", _fmt_pct_from_frac(r_meas, 1, with_symbol=True)])

    r_pim = report.get_float("R_PIM")
    if r_pim is not None:
        rows.append(["R-pim", _fmt_pct_from_frac(r_pim, 1, with_symbol=True)])

    cc_half = report.get_float("CC_HALF")
    if cc_half is not None:
        rows.append(["CC\u00bd\u00b3", _fmt_pct_from_frac(cc_half, 1, with_symbol=True)])

    cc_star = report.get_float("CC_STAR")
    if cc_star is not None:
        rows.append(["CC*", _fmt_pct_from_frac(cc_star, 1, with_symbol=True)])

    isa_raw = report.get_str("ISA")
    if isa_raw is not None:
        isa_val = report.get_float("ISA")
        if isa_val is not None:
            isa_low = report.get_float("ISA_LOW")
            isa_high = report.get_float("ISA_HIGH")
            if isa_low is not None and isa_high is not None:
                isa_display = f"{isa_val:.1f} ({isa_low:.1f}-{isa_high:.1f})"
            else:
                isa_display = f"{isa_val:.1f}"
        else:
            isa_display = isa_raw
        rows.append(["ISa\u2074", isa_display])

    wilson_b = report.get_float("WILSON_B")
    if wilson_b is not None:
        rows.append(["Wilson B (A\u00b2)", f"{wilson_b:.2f}"])

    sig_ano = report.get_float("SIGANO")
    if sig_ano is not None:
        rows.append(["SigAno", f"{sig_ano:.2f}"])

    cc_anom = report.get_float("CC_ANOM")
    if cc_anom is not None:
        rows.append(["CC_anom", _fmt_pct_from_frac(cc_anom, 1, with_symbol=True)])

    # Footnotes
    written_note = ""
    if len(res_range) >= 2:
        written_note = f" Written reflection range: {res_range[0]:.2f}-{res_range[1]:.2f} A."
    sg_alt = report.get_str("SPACE_GROUP_ALTERNATIVES")
    sg_alt_note = f" Indistinguishable alternatives: {sg_alt}." if sg_alt and sg_alt != "NONE" else ""
    rugnux_ver = report.get_str("RUGNUX_VERSION") or "unknown"

    notes = (
        "1. Heuristic Data Quality Score (0.0-1.0) combining resolution, completeness, I/Sigma(I), "
        "CC1/2, R-meas, ISa, and rugnux pathology verdict.\n"
        f"2. Space group determined by rugnux ({report.get_str('SPACE_GROUP_SEARCH') or 'auto'}).{sg_alt_note}\n"
        "3. Percentage correlation between intensities from random half-datasets "
        "(Karplus & Diederichs (2012), Science 336: 1030-1033).\n"
        "4. Asymptotic I/Sigma(I) limit of the experimental setup "
        "(Diederichs (2010), Acta Cryst D66: 733-740).\n"
        f"5. Fitted resolution where CC1/2 fall-off crosses 0.30.{written_note} "
        f"Processed with rugnux {rugnux_ver}."
    )

    return {
        "title": "Data Quality Statistics",
        "kind": "table",
        "data": rows,
        "header": "column",
        "notes": notes,
    }


def _build_symmetry_table(report: RugnuxReport) -> dict[str, Any] | None:
    rows: list[list[Any]] = [["Property", "Value"]]

    sg_name = report.get_str("SPACE_GROUP_NAME")
    sg_num = report.get_int("SPACE_GROUP_NUMBER")
    if sg_name:
        sg_label = f"{sg_name} (#{sg_num})" if sg_num is not None else sg_name
        rows.append(["Selected Space Group", sg_label])

    sg_alt = report.get_str("SPACE_GROUP_ALTERNATIVES")
    if sg_alt:
        rows.append(["Indistinguishable Alternatives", sg_alt])

    sg_achiral = report.get_str("SPACE_GROUP_ACHIRAL_ALTERNATIVES")
    if sg_achiral:
        rows.append(["Achiral Alternatives", sg_achiral])

    sohncke = report.get_str("SOHNCKE_SPACE_GROUP")
    if sohncke:
        rows.append(["Sohncke Space Group", sohncke])

    enantiomorph = report.get_str("SPACE_GROUP_ENANTIOMORPH")
    if enantiomorph:
        rows.append(["Enantiomorph Hand", enantiomorph])

    sg_search = report.get_str("SPACE_GROUP_SEARCH")
    if sg_search:
        rows.append(["Space Group Search", sg_search])

    setting_op = report.get_str("SETTING_OPERATOR")
    setting_src = report.get_str("SETTING_SOURCE")
    if setting_op or setting_src:
        setting_val = f"{setting_op or '-'} ({setting_src or '-'})"
        rows.append(["Setting", setting_val])

    idx_ambig = report.get_str("INDEXING_AMBIGUITY_OPERATORS")
    rows.append(["Indexing Ambiguity Operators", idx_ambig if idx_ambig else "None"])

    cell_esd = report.get_str("CELL_ESD")
    if cell_esd:
        rows.append(["Unit Cell ESD", cell_esd])

    centre_verdict = report.get_str("CENTRE_STATISTICS_VERDICT")
    if centre_verdict:
        rows.append(["Intensity Distribution", centre_verdict])

    if len(rows) <= 2:  # Only header + default Indexing Ambiguity row
        return None

    return {
        "title": "Space Group & Symmetry",
        "kind": "table",
        "data": rows,
        "header": "row",
    }


def _build_geometry_table(report: RugnuxReport) -> dict[str, Any] | None:
    rows: list[list[Any]] = [["Parameter", "Value"]]

    def add(label: str, key: str, unit: str = "") -> None:
        val = report.get_str(key)
        if val is not None:
            rows.append([label, f"{val}{unit}"])

    add("Experiment Type", "EXPERIMENT_TYPE")
    add("Images Processed", "IMAGES_PROCESSED")
    add("Wavelength", "WAVELENGTH", " A")
    add("Oscillation Range", "OSCILLATION_RANGE", " deg")
    add("Starting Angle", "STARTING_ANGLE", " deg")
    add("Rotation Axis", "ROTATION_AXIS")
    add("Detector Distance", "DETECTOR_DISTANCE", " mm")
    add("Beam Centre (PONI)", "BEAM_CENTRE", " px")
    add("Direct Beam (Refined)", "DIRECT_BEAM", " px")
    add("Direct Beam (Input Header)", "FILE_DIRECT_BEAM", " px")

    shift = report.get_float("BEAM_CENTRE_SHIFT")
    tol = report.get_float("BEAM_CENTRE_TOLERANCE")
    if shift is not None:
        if tol is not None:
            rows.append(["Beam Centre Shift", f"{shift:.2f} px (tolerance {tol:.2f} px)"])
        else:
            rows.append(["Beam Centre Shift", f"{shift:.2f} px"])

    add("Detector Tilt (rot1/rot2/rot3)", "DETECTOR_TILT", " deg")
    add("Flight Path", "FLIGHT_PATH")
    add("Spot Resolution Estimate", "SPOT_RESOLUTION_ESTIMATE", " A")

    wall_time = report.get_str("WALL_TIME")
    gpu = report.get_str("GPU")
    if wall_time is not None:
        runtime_str = f"{wall_time} s" + (f" ({gpu})" if gpu else "")
        rows.append(["Processing Wall Time", runtime_str])

    if len(rows) <= 1:
        return None

    return {
        "title": "Experiment & Refined Geometry",
        "kind": "table",
        "data": rows,
        "header": "row",
        "notes": (
            "Post-refined geometry used for final integration. Beam Centre is the PONI "
            "(point of normal incidence); Direct Beam is where the beam intersects the detector."
        ),
    }


def _build_frame_section(report: RugnuxReport, plot: PlotTable | None) -> dict[str, Any] | None:
    if plot is None or plot.n_frames == 0:
        return None

    frames = plot.frame_numbers
    content: list[dict[str, Any]] = []

    def add_scatter(
        y1_label: str,
        y1_col: str,
        y2_label: str | None = None,
        y2_col: str | None = None,
    ) -> None:
        y1_vals = plot.get_column(y1_col)
        y2_vals = plot.get_column(y2_col) if y2_col else []
        has_y1 = _has_non_null(y1_vals)
        has_y2 = bool(y2_label and _has_non_null(y2_vals))
        if not has_y1 and not has_y2:
            return

        data: dict[str, Any] = {"x": ["Frame Number", *frames]}
        if has_y1:
            data["y1"] = [[y1_label, *y1_vals]]
            data["y2"] = [[y2_label, *y2_vals]] if (has_y2 and y2_label) else []
        elif has_y2 and y2_label:
            data["y1"] = [[y2_label, *y2_vals]]
            data["y2"] = []

        content.append(
            {
                "kind": "scatterplot",
                "style": "half-height",
                "data": data,
            }
        )

    add_scatter("Scale Factor", "scale_G", "Mosaicity (deg)", "sigma_M_deg")
    add_scatter("CC to Merge", "cc_to_merge", "Spots", "spots")
    add_scatter("Resolution (A)", "resolution_A", "Background", "bkg")

    if plot.has_non_merged_disposition:
        add_scatter("Disposition", "disposition")

    if not content:
        return None

    note_parts: list[str] = []
    merged_n = report.get_int("FRAMES_MERGED")
    downgraded_n = report.get_int("FRAMES_DOWNGRADED")
    rejected_n = report.get_int("FRAMES_REJECTED")
    if merged_n is not None:
        note_parts.append(
            f"* Frames: {merged_n} merged, {downgraded_n or 0} downgraded, {rejected_n or 0} rejected."
        )
    rad_b = report.get_float("RADIATION_DAMAGE_RELATIVE_B")
    if rad_b is not None:
        note_parts.append(f"* Radiation damage: relative B-factor change {rad_b:+.2f} A\u00b2 over sweep.")
    note_parts.append(
        "* Mosaicity (sigma_M_deg) is the smoothed per-frame Gaussian rocking-curve standard deviation."
    )
    if plot.has_non_merged_disposition:
        note_parts.append("* Disposition: 1 = merged, 0.5 = downgraded, 0 = rejected.")

    content[-1]["notes"] = "\n".join(note_parts)

    return {
        "title": "Statistics of Intensities by Frame Number",
        "content": content,
    }


def _shell_table_row_primary(row: ShellRow) -> list[Any]:
    shell_val: Any = row.label if row.is_overall else row.d_min
    return [
        shell_val,
        _fmt_int(row.n_obs),
        _fmt_int(row.n_uniq),
        _fmt_float(row.completeness, 1),
        _fmt_float(row.multiplicity, 1),
        _fmt_float(row.r_meas, 1),
        _fmt_float(row.r_pim, 1),
        _fmt_float(row.cc_half, 1),
        _fmt_float(row.cc_star, 1),
        _fmt_float(row.i_over_sigma, 1),
        _fmt_float(row.sig_ano, 2),
        _fmt_float(row.cc_anom, 1),
    ]


def _shell_table_row_anom(row: ShellRow) -> list[Any]:
    shell_val: Any = row.label if row.is_overall else row.d_min
    return [
        shell_val,
        _fmt_int(row.n_obs),
        _fmt_int(row.n_uniq),
        _fmt_float(row.completeness, 1),
        _fmt_float(row.multiplicity, 1),
        _fmt_float(row.r_meas, 1),
        _fmt_float(row.cc_half, 1),
        _fmt_float(row.cc_star, 1),
        _fmt_float(row.i_over_sigma, 1),
        _fmt_float(row.sig_ano, 2),
        _fmt_float(row.cc_anom, 1),
    ]


def _build_shell_section(report: RugnuxReport) -> dict[str, Any] | None:
    if not report.is_merged or not report.primary_shells:
        return None

    shells = [s for s in report.primary_shells if s.d_min is not None]
    if not shells:
        return None

    x_shells = [s.d_min for s in shells]
    content: list[dict[str, Any]] = []

    # Lineplot 1: Completeness (%) + CC1/2 (%) vs R-meas (%)
    compl_vals = [s.completeness for s in shells]
    cchalf_vals = [s.cc_half for s in shells]
    rmeas_vals = [s.r_meas for s in shells]

    y1_series: list[list[Any]] = []
    if _has_non_null(compl_vals):
        y1_series.append(["Completeness (%)", *compl_vals])
    if _has_non_null(cchalf_vals):
        y1_series.append(["CC\u00bd (%)", *cchalf_vals])

    if y1_series:
        plot1_data: dict[str, Any] = {
            "x": ["Resolution Shell", *x_shells],
            "y1": y1_series,
            "y2": [],
            "x-scale": "inv-square",
        }
        if _has_non_null(rmeas_vals):
            plot1_data["y2"] = [["R-meas (%)", *rmeas_vals]]
            plot1_data["y2-label"] = "R-factors (%)"
        content.append({"kind": "lineplot", "data": plot1_data})

    # Lineplot 2: I/Sigma(I) vs SigAno
    ios_vals = [s.i_over_sigma for s in shells]
    sigano_vals = [s.sig_ano for s in shells]
    if _has_non_null(ios_vals):
        plot2_data: dict[str, Any] = {
            "x": ["Resolution Shell", *x_shells],
            "y1": [["I/Sigma(I)", *ios_vals]],
            "y2": [["SigAno", *sigano_vals]] if _has_non_null(sigano_vals) else [],
            "x-scale": "inv-square",
        }
        content.append({"kind": "lineplot", "data": plot2_data})

    # Primary Shell Table
    primary_rows: list[list[Any]] = [
        [
            "Shell (A)",
            "N_obs",
            "N_uniq",
            "Completeness (%)",
            "Mult",
            "R_meas (%)",
            "R_pim (%)",
            "CC\u00bd (%)",
            "CC* (%)",
            "I/Sigma(I)\u00b9",
            "SigAno\u00b2",
            "CC_anom (%)\u00b3",
        ]
    ]
    for s in shells:
        primary_rows.append(_shell_table_row_primary(s))
    if report.primary_overall is not None:
        primary_rows.append(_shell_table_row_primary(report.primary_overall))

    note_lines = [
        "1. Mean intensity/Sigma(I) of unique reflections after merging.",
        "2. Mean anomalous difference in units of its estimated standard deviation.",
        "3. Half-set correlation of anomalous differences (CC_anom).",
    ]
    if report.wilson_note:
        note_lines.append(f"* {report.wilson_note}")
    err_a = report.get_str("ERROR_MODEL_A")
    err_b = report.get_str("ERROR_MODEL_B")
    if err_a and err_b:
        note_lines.append(f"* Error model: a = {err_a}, b = {err_b}.")

    content.append(
        {
            "title": "Merging Statistics by Resolution Shell",
            "kind": "table",
            "data": primary_rows,
            "header": "row",
            "notes": "\n".join(note_lines),
        }
    )

    # Conditional Anomalous Shell Table
    if report.anom_shells and report.has_anomalous_signal():
        anom_rows: list[list[Any]] = [
            [
                "Shell (A)",
                "N_obs",
                "N_uniq",
                "Completeness (%)",
                "Mult",
                "R_meas (%)",
                "CC\u00bd (%)",
                "CC* (%)",
                "I/Sigma(I)",
                "SigAno",
                "CC_anom (%)",
            ]
        ]
        for s in report.anom_shells:
            if s.d_min is not None:
                anom_rows.append(_shell_table_row_anom(s))
        if report.anom_overall is not None:
            anom_rows.append(_shell_table_row_anom(report.anom_overall))

        content.append(
            {
                "title": "Anomalous Statistics by Resolution Shell (Friedel's Law Off)",
                "kind": "table",
                "data": anom_rows,
                "header": "row",
                "notes": (
                    "Same merged data counted with Friedel's law OFF (acentric Bijvoet mates counted separately)."
                ),
            }
        )

    return {
        "title": "Statistics of Final Reflections by Shell",
        "content": content,
    }


def _build_diagnostics_section(report: RugnuxReport) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    # 1. Verdict & Warnings description block
    verdict_lines: list[str] = [f"Verdict: {report.verdict}"]
    verdict_text = report.get_str("VERDICT_TEXT")
    if verdict_text:
        verdict_lines.append(verdict_text)
    if report.pathology_flags:
        verdict_lines.append(f"Pathology flags: {', '.join(report.pathology_flags)}")
    if report.advice_codes:
        verdict_lines.append(f"Advice: {', '.join(report.advice_codes)}")
    if report.warnings:
        verdict_lines.append("Warnings:")
        for w in report.warnings:
            verdict_lines.append(f"- {w}")
    else:
        verdict_lines.append("Warnings: None.")
    if report.advice_lines:
        verdict_lines.append("Collection advice:")
        for a in report.advice_lines:
            verdict_lines.append(f"- {a}")

    content.append(
        {
            "title": "Verdict & Warnings",
            "description": "\n".join(verdict_lines),
        }
    )

    # 2. Diagnostics summary table
    diag_rows: list[list[Any]] = [["Check", "Status", "Details"]]

    if report.get_int("OVERLOADED_REFLECTIONS") is not None:
        ov_n = report.get_int("OVERLOADED_REFLECTIONS")
        ov_pct = report.get_float("OVERLOADED_REFLECTIONS_PCT")
        ov_lr = report.get_float("OVERLOADED_REFLECTIONS_LOW_RES_PCT")
        flagged = "OVERLOADS" in report.pathology_flags
        diag_rows.append(
            [
                "Overloads",
                "WARNING" if flagged else "OK",
                f"{ov_n} reflections ({_fmt_float(ov_pct, 3)}%; {_fmt_float(ov_lr, 3)}% coarser than 4 A)",
            ]
        )

    if report.get_float("OVERLAPPED_REFLECTIONS_PCT") is not None:
        ol_pct = report.get_float("OVERLAPPED_REFLECTIONS_PCT")
        ol_maj = report.get_float("OVERLAPPED_REFLECTIONS_MAJOR_PCT")
        flagged = "OVERLAPS" in report.pathology_flags
        diag_rows.append(
            [
                "Overlaps",
                "WARNING" if flagged else "OK",
                f"{_fmt_float(ol_pct, 2)}% share signal pixels ({_fmt_float(ol_maj, 2)}% major)",
            ]
        )

    if report.get_bool("ICE_RINGS_DETECTED") is not None:
        ice_det = report.get_bool("ICE_RINGS_DETECTED")
        ice_z = report.get_str("ICE_INTENSITY_Z")
        ice_rings = report.get_int("ICE_INTENSITY_RINGS")
        flagged = bool(ice_det) or ("ICE_RINGS" in report.pathology_flags)
        detail = f"Image rings: {'detected' if ice_det else 'none'}"
        if ice_z is not None:
            detail += f"; intensity core-flank contrast {ice_z} sigma"
            if ice_rings is not None:
                detail += f" ({ice_rings} rings)"
        diag_rows.append(["Ice Rings", "WARNING" if flagged else "OK", detail])

    if report.get_bool("POWDER_RINGS_DETECTED") is not None:
        powder = report.get_bool("POWDER_RINGS_DETECTED")
        diag_rows.append(
            [
                "Powder Rings",
                "WARNING" if powder else "OK",
                "Detected" if powder else "None detected",
            ]
        )

    if report.get_bool("MULTIPLE_LATTICES") is not None:
        multi = report.get_bool("MULTIPLE_LATTICES")
        main_spots = report.get_float("MAIN_LATTICE_SPOTS_PCT")
        main_int = report.get_float("MAIN_LATTICE_INTENSITY_PCT")
        extra_n = report.get_int("EXTRA_LATTICE_COUNT", 0)
        detail = (
            f"Main lattice: {_fmt_float(main_spots, 1)}% spots, "
            f"{_fmt_float(main_int, 1)}% intensity; {extra_n} extra lattice(s)"
        )
        diag_rows.append(["Multiple Lattices", "WARNING" if multi else "OK", detail])

    if report.get_bool("SUPERCELL_POSSIBLE") is not None:
        sc = report.get_bool("SUPERCELL_POSSIBLE")
        sc_cls = report.get_str("SUPERCELL_CLASS") or "-"
        sc_occ = report.get_float("SUPERCELL_OCCUPANCY_PCT")
        sc_rock = report.get_float("SUPERCELL_ROCK_PCT")
        detail = f"Class ({sc_cls}): occupancy {_fmt_float(sc_occ, 1)}%, rock {_fmt_float(sc_rock, 1)}%"
        diag_rows.append(["Supercell", "WARNING" if sc else "OK", detail])

    if report.get_bool("HARMONIC_CONTAMINATION") is not None:
        harm = report.get_bool("HARMONIC_CONTAMINATION")
        h_spot = report.get_float("HARMONIC_SPOT_PCT")
        h_null = report.get_float("HARMONIC_NULL_PCT")
        diag_rows.append(
            [
                "Harmonic Contamination",
                "WARNING" if harm else "OK",
                f"Spots {_fmt_float(h_spot, 1)}% vs null {_fmt_float(h_null, 1)}%",
            ]
        )

    tncs = report.get_str("TNCS_DETECTED")
    if tncs is not None:
        peak_pct = report.get_float("TNCS_PATTERSON_PEAK_PCT")
        null_pct = report.get_str("TNCS_PATTERSON_NULL_PCT")
        vec = report.get_str("TNCS_VECTOR")
        detail = f"Patterson peak {_fmt_float(peak_pct, 1)}%"
        if null_pct:
            detail += f" (null {null_pct}%)"
        if tncs in {"TRUE", "POSSIBLE"} and vec:
            detail += f", vector ({vec})"
        status = "WARNING" if tncs in {"TRUE", "POSSIBLE"} else ("OK" if tncs == "FALSE" else tncs)
        diag_rows.append(["Translational NCS", status, detail])

    twin_verdict = report.get_str("TWINNING_VERDICT")
    if twin_verdict is not None:
        l_mean = report.get_float("L_TEST_MEAN_ABS_L")
        sec_mom = report.get_float("SECOND_MOMENT_I")
        twin_frac = report.get_float("ESTIMATED_TWIN_FRACTION")
        detail = f"<|L|> = {_fmt_float(l_mean, 3)}, <I\u00b2>/<I>\u00b2 = {_fmt_float(sec_mom, 3)}"
        if twin_frac is not None:
            detail += f", est. twin fraction {_fmt_float(twin_frac, 3)}"
        status = "OK" if twin_verdict == "NO_INDICATION" else twin_verdict
        diag_rows.append(["Twinning", status, detail])

    rad_b = report.get_float("RADIATION_DAMAGE_RELATIVE_B")
    if rad_b is not None:
        diag_rows.append(["Radiation Damage", "INFO", f"Relative B change: {rad_b:+.2f} A\u00b2"])

    aniso_v = report.get_str("ANISOTROPY_VERDICT")
    if aniso_v is not None:
        delta_b = report.get_float("ANISOTROPY_DELTA_B")
        d_prin = report.get_str("ANISOTROPY_D_MIN_PRINCIPAL")
        detail = f"\u0394B = {_fmt_float(delta_b, 2)} A\u00b2"
        if d_prin:
            detail += f"; principal d_min: {d_prin} A"
        status = "OK" if aniso_v == "NOT_DETECTED" else ("WARNING" if "ANISOTROPY" in report.pathology_flags else aniso_v)
        diag_rows.append(["Anisotropy", status, detail])

    cov = report.get_float("SWEEP_COVERAGE")
    if cov is not None:
        cov_full = report.get_float("SWEEP_COVERAGE_FULL_TURN")
        blind = report.get_float("SPINDLE_LOST_UNIQUE_FRACTION")
        detail = f"Coverage {cov:.1f}%"
        if cov_full is not None:
            detail += f" (full turn {cov_full:.1f}%)"
        if blind is not None:
            detail += f"; blind-cone loss {blind * 100.0:.2f}%"
        diag_rows.append(["Sweep Coverage", "OK", detail])

    if len(diag_rows) > 1:
        content.append(
            {
                "title": "Data Quality Diagnostics",
                "kind": "table",
                "data": diag_rows,
                "header": "row",
            }
        )

    # 3. Twin Laws description block
    twin_lines = list(report.twin_laws)
    sub_sg = report.get_str("TWIN_SUBGROUP_SPACE_GROUP")
    if sub_sg:
        sub_law = report.get_str("TWIN_SUBGROUP_LAW") or "-"
        sub_frac = report.get_str("TWIN_SUBGROUP_FRACTION") or "-"
        twin_lines.append(
            f"Twin subgroup hypothesis: {sub_sg} (law {sub_law}, fraction {sub_frac})."
        )

    content.append(
        {
            "title": "Twin Laws",
            "description": "\n".join(twin_lines)
            if twin_lines
            else "No twin laws are possible for this crystal lattice.",
        }
    )

    return {
        "title": "Diagnostics & Warnings",
        "content": content,
    }


def _build_sweeps_section(report: RugnuxReport) -> dict[str, Any] | None:
    if not report.sweeps:
        return None

    rows: list[list[Any]] = [
        [
            "Sweep",
            "Input",
            "Images",
            "Included",
            "Decision",
            "Resolution (A)",
            "Completeness (%)",
            "CC\u00bd",
            "Scale",
            "Relative B",
            "ISa",
            "Reindex Op",
        ]
    ]
    for sw in report.sweeps:
        f = sw.fields
        inp = f.get("INPUT", "-")
        rows.append(
            [
                sw.index,
                Path(inp).name if inp != "-" else "-",
                f.get("IMAGES", "-"),
                f.get("INCLUDED", "-"),
                f.get("DECISION", "-"),
                f.get("RESOLUTION_ALONE", "-"),
                f.get("COMPLETENESS_ALONE", "-"),
                f.get("CC_HALF_ALONE", "-"),
                f.get("SCALE", "-"),
                f.get("RELATIVE_B", "-"),
                f.get("ISA", "-"),
                f.get("REINDEX_OP", "-"),
            ]
        )

    notes_parts: list[str] = []
    ref = report.get_str("SWEEP_REFERENCE")
    decay = report.get_str("SWEEP_DECAY_MODEL")
    if ref:
        notes_parts.append(f"Reference sweep: #{ref}.")
    if decay:
        notes_parts.append(f"Decay model: {decay}.")

    table: dict[str, Any] = {
        "title": "Sweeps in Joint Merge",
        "kind": "table",
        "data": rows,
        "header": "row",
    }
    if notes_parts:
        table["notes"] = " ".join(notes_parts)

    return {
        "title": "Multi-Sweep Summary",
        "content": [table],
    }


def _build_model_section(report: RugnuxReport) -> dict[str, Any] | None:
    model_keys = (
        ("Model Validation", "MODEL_VALIDATION"),
        ("Validation Reason", "MODEL_VALIDATION_REASON"),
        ("Placement", "MODEL_PLACEMENT"),
        ("Fit Verdict", "MODEL_FIT"),
        ("Fit Sigma", "MODEL_FIT_SIGMA"),
        ("R-work", "R_WORK"),
        ("R-free", "R_FREE"),
        ("Refined R-work", "REFINE_R_WORK"),
        ("Refined R-free", "REFINE_R_FREE"),
        ("R-free Flags", "R_FREE_FLAGS"),
        ("Decisions Taken", "MODEL_DECISIONS_TAKEN"),
        ("Enantiomorph Adopted", "MODEL_ENANTIOMORPH_ADOPTED"),
        ("Indexing Operator", "MODEL_INDEXING_OPERATOR"),
        ("Anomalous Scatterer Mean Sigma", "ANOMALOUS_SCATTERER_MEAN_SIGMA"),
    )
    rows: list[list[Any]] = [["Metric", report.dataset_name()]]
    for label, key in model_keys:
        val = report.get_str(key)
        if val is not None:
            rows.append([label, val])

    if len(rows) <= 1:
        return None

    return {
        "title": "Model Validation & Refinement",
        "content": [
            {
                "title": "Model Validation Statistics",
                "kind": "table",
                "data": rows,
                "header": "column",
            }
        ],
    }


def build_report(
    report: RugnuxReport,
    plot: PlotTable | None = None,
    *,
    directory: str | Path | None = None,
    filename: str = "report.json",
    report_id: Any = None,
    data_id: Any = None,
) -> dict[str, Any]:
    """Build an MxLIVE/AutoProcess-compatible report.json dict from parsed rugnux outputs."""
    score = compute_score(report)
    dataset = report.dataset_name()

    if directory is not None:
        dir_str = str(Path(directory).resolve())
    elif report.source_path is not None:
        dir_str = str(report.source_path.resolve().parent)
    else:
        dir_str = str(Path.cwd().resolve())

    # Section 1: Data Quality Summary
    summary_content: list[dict[str, Any]] = [_build_summary_stats_table(report, score)]
    sym_table = _build_symmetry_table(report)
    if sym_table is not None:
        summary_content.append(sym_table)
    geom_table = _build_geometry_table(report)
    if geom_table is not None:
        summary_content.append(geom_table)

    details: list[dict[str, Any]] = [
        {
            "title": "Data Quality Summary",
            "content": summary_content,
        }
    ]

    # Section 2: Per-frame plots
    frame_section = _build_frame_section(report, plot)
    if frame_section is not None:
        details.append(frame_section)

    # Section 3: Resolution shell plots & tables
    shell_section = _build_shell_section(report)
    if shell_section is not None:
        details.append(shell_section)

    # Section 4: Diagnostics & Warnings
    details.append(_build_diagnostics_section(report))

    # Optional Section 5: Multi-sweep summary
    sweeps_section = _build_sweeps_section(report)
    if sweeps_section is not None:
        details.append(sweeps_section)

    # Optional Section 6: Model validation & refinement
    model_section = _build_model_section(report)
    if model_section is not None:
        details.append(model_section)

    kind = "MX Anomalous Analysis" if report.is_anomalous_run else "MX Native Analysis"

    return {
        "id": report_id,
        "directory": dir_str,
        "filename": filename,
        "score": score,
        "data_id": data_id,
        "title": f'Analysis of "{dataset}"',
        "kind": kind,
        "details": details,
    }
