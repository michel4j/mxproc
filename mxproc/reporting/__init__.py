import json
import shutil
from pathlib import Path
from typing import List

from mxproc import Analysis
from mxproc.xtal import Experiment
from mxproc.common import StepType, load_json

from .generic import summary_table, lattice_table, spacegroup_table, standard_error_report
from .generic import shell_statistics_report, frame_statistics_report, wilson_report, twinning_report
from .screening import screening_summary, screening_strategy, screening_completeness
from .text import text_report


DATA_DIR = Path(__file__).parent / 'data'


def save_report(report: dict, path: Path):
    """
    Save the given report to the
    :param report: dictionary containing the report
    :param path: location to store the report
    """

    report_file = path / 'report.json'
    text_file = path / 'report.txt'
    report.update(id=None, data_id=None, directory=str(path), filename='report.json')

    # read previous json_file and obtain id from it if one exists:
    if report_file.exists():
        old_report = load_json(report_file)
        report['id'] = old_report.get('id')
        report['data_id'] = old_report.get('data_id')

    # save
    with open(report_file, 'w') as file:
        json.dump(report, file, indent=4)

    with open(text_file, 'w') as handle:
        handle.write(text_report(report['details']))

    shutil.copy(DATA_DIR / 'report.html', path)
    shutil.copy(DATA_DIR / 'report.min.js', path)
    shutil.copy(DATA_DIR / 'report.min.css', path)


def screening_report(analysis) -> List[dict]:
    """
    Prepare a screening report
    :param analysis: analysis object
    :return: list of dictionaries
    """
    experiment = None
    for expt in analysis.experiments:
        if analysis.get_step_result(expt, StepType.STRATEGY):
            experiment = expt
            break

    if experiment is None:
        return []
    else:
        indexing = analysis.get_step_result(experiment, StepType.INDEX)
        strategy = analysis.get_step_result(experiment, StepType.STRATEGY)
        return [
            {
                'title': f'Screening Report for Dataset "{experiment.name}"',
                'content': [
                    screening_summary(analysis, experiment),
                    screening_strategy(strategy),
                    lattice_table(indexing),
                    screening_completeness(strategy),
                ]
            }
        ]


def single_details(analysis: Analysis, experiment: Experiment) -> List[dict]:
    anom = "Anomalous" if analysis.options.anomalous else "Native"

    indexing = analysis.get_step_result(experiment, StepType.INDEX)
    integration = analysis.get_step_result(experiment, StepType.INTEGRATE)
    symmetry = analysis.get_step_result(experiment, StepType.SYMMETRY)
    scaling = analysis.get_step_result(experiment, StepType.SCALE)

    return [
        {
            'title': f"{anom} Data Analysis Report for '{experiment.name}'",
            'content': [
                summary_table(analysis),
                lattice_table(integration),
                spacegroup_table(symmetry)
            ],
        },
        standard_error_report(integration),
        frame_statistics_report(symmetry),
        shell_statistics_report(scaling),
        wilson_report(scaling),
        twinning_report(scaling),
    ]


def merging_details(analysis: Analysis):

    report_list = []
    for experiment in analysis.experiments:
        report_list.extend(single_details(analysis, experiment))

    return report_list


def multi_details(analysis: Analysis):
    report_list = []
    for experiment in analysis.experiments:
        report_list.extend(single_details(analysis, experiment))

    return report_list


