# Engine Report Builders Converge at `save_report`

Engines produce fundamentally different diagnostic data: XDS assembles a typed `ReportPayload` consumed by `mxproc.reporting.generic` (whose tables and footnotes reflect XDS, POINTLESS, XDSSTAT, and Xtriage outputs), whereas Rugnux produces native pathology verdicts, multi-experiment summaries, and model-validation sections that do not fit XDS's schema. We decided that engines converge at the MxLIVE/AutoProcess report document contract (`reporting.save_report`) rather than forcing every engine through `ReportPayload`.

## Considered Options

- **Force all engines onto `ReportPayload`**: Rejected because `ReportPayload` and `reporting/generic.py` encode XDS-specific intermediate concepts and auxiliary-program footnotes; extending it to cover Rugnux (and future engines like DIALS) would either discard engine-native diagnostics or bloat `ReportPayload` into a fragile union.
- **Move `reporting/generic.py` and `ReportPayload` into XDS immediately**: Deferred; `ReportPayload` remains in `mxproc.reporting` for engines that share the multi-step indexing/integration/symmetry/scaling report layout, while `save_report` and the text/HTML renderers are shared across all engines.

## Consequences

- Engines may either build a `ReportPayload` for `reporting.generate_reports()` or assemble an MxLIVE-compatible report dictionary directly and call `reporting.save_report()`.
- `mxproc.reporting` remains decoupled from engine step definitions while owning the shared persistence and rendering path (`report.json`, `report.txt`, `report.html`).
