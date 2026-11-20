import json
import shutil
from pathlib import Path
from typing import List

from mxproc.command import CommandFailed
from mxproc.common import load_json

from .generic import (
    frame_statistics_report,
    lattice_table,
    shell_statistics_report,
    spacegroup_table,
    standard_error_report,
    summary_table,
    twinning_report,
    wilson_report,
)
from .models import (
    ExperimentReportData,
    IndexReportData,
    IntegrationReportData,
    QualitySummary,
    ReportPayload,
    ScalingReportData,
    StrategyReportData,
    SymmetryReportData,
    TwinningReportData,
    build_report_payload,
)
from .screening import screening_completeness, screening_strategy, screening_summary
from .text import text_report

DATA_DIR = Path(__file__).parent / 'data'


def save_report(report: dict, path: Path, outputs: dict = None):
    """
    Save the given report to disk as report.json, report.txt, and report.html.
    """
    outputs = outputs or {}
    report_file = path / 'report.json'
    text_file = path / 'report.txt'
    report.update(id=None, data_id=None, directory=str(path), outputs=outputs, filename='report.json')

    if report_file.exists():
        old_report = load_json(report_file)
        report['id'] = old_report.get('id')
        report['data_id'] = old_report.get('data_id')

    with open(report_file, 'w') as file:
        json.dump(report, file, indent=4)

    with open(text_file, 'w') as handle:
        handle.write(text_report(report['details']))

    shutil.copy(DATA_DIR / 'report.html', path)
    shutil.copy(DATA_DIR / 'report.min.js', path)
    shutil.copy(DATA_DIR / 'report.min.css', path)


def screening_report(expt: ExperimentReportData) -> List[dict]:
    """
    Prepare a screening report for a single experiment.
    """
    if expt.indexing is None or expt.strategy is None:
        return []

    return [
        {
            'title': f'Screening Report for Dataset "{expt.name}"',
            'content': [
                screening_summary(expt),
                screening_strategy(expt.strategy),
                lattice_table(expt.indexing.lattices),
                screening_completeness(expt.strategy),
            ],
        }
    ]


def single_details(payload: ReportPayload, expt: ExperimentReportData) -> List[dict]:
    anom = "Anomalous" if payload.anomalous else "Native"
    assert expt.integration is not None
    assert expt.symmetry is not None
    assert expt.scaling is not None

    content = [summary_table(payload)]
    if expt.integration.lattices:
        content.append(lattice_table(expt.integration.lattices))
    if expt.symmetry.candidates:
        content.append(spacegroup_table(expt.symmetry))

    details = [
        {
            'title': f"{anom} Data Analysis Report for '{expt.name}'",
            'content': content,
        }
    ]
    if expt.integration.errors[:-1]:
        details.append(standard_error_report(expt.integration))
    if expt.symmetry.frames or expt.symmetry.differences:
        details.append(frame_statistics_report(expt.symmetry))
    if expt.scaling.statistics:
        details.append(shell_statistics_report(expt.scaling))
    if expt.scaling.twinning.plots:
        details.append(wilson_report(expt.scaling.twinning))
    details.append(twinning_report(expt.scaling.twinning))
    return details


def merging_details(payload: ReportPayload) -> List[dict]:
    report_list = []
    for expt in payload.experiments:
        if expt.scaling is not None:
            report_list.extend(single_details(payload, expt))
    return report_list


def generate_reports(payload: ReportPayload) -> List[Path]:
    """
    Build and write reports according to payload mode (screen / merge / single).
    Returns paths to generated report.html files.
    """
    anom = 'Anomalous' if payload.anomalous else 'Native'
    generated: List[Path] = []

    if payload.screen:
        screened = [expt for expt in payload.experiments if expt.strategy is not None]
        if not screened:
            raise CommandFailed('No successful results to report on!')

        name_label = ", ".join(expt.name for expt in screened)
        first = screened[0].strategy
        assert first is not None
        report = {
            'title': f"{anom} Screening of {name_label}",
            'kind': 'MX Screening',
            'score': round(first.score, 2),
            'strategy': first.strategy,
            'details': screening_report(screened[0]),
        }
        save_report(report, payload.directory)
        generated.append(payload.directory / 'report.html')
        return generated

    scaled = [expt for expt in payload.experiments if expt.scaling is not None]
    if not scaled:
        raise CommandFailed('No successful results to report on!')

    if payload.merge:
        first_scaling = scaled[0].scaling
        assert first_scaling is not None
        name_label = ", ".join(expt.name for expt in scaled)
        report = {
            'title': f"{anom} Merging of {name_label}",
            'kind': f'MX {anom} Analysis',
            'score': round(first_scaling.summary.score, 2),
            'details': merging_details(payload),
        }
        save_report(report, payload.directory, outputs=payload.outputs)
        generated.append(payload.directory / 'report.html')
    else:
        for expt in scaled:
            assert expt.scaling is not None
            report = {
                'title': f"{anom} Analysis of {expt.name}",
                'kind': f'MX {anom} Analysis',
                'score': round(expt.scaling.summary.score, 2),
                'details': single_details(payload, expt),
            }
            save_report(report, expt.work_dir, outputs=payload.outputs)
            generated.append(expt.work_dir / 'report.html')

    return generated
