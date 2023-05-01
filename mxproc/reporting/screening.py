import inspect
from collections import defaultdict

from mxproc import Analysis
from mxproc.common import StepType
from mxproc.xtal import Experiment, Lattice


def get_strategy(results):
    strategy = results['strategy']
    run = strategy['runs'][0]
    info = {
        'attenuation': strategy['attenuation'],
        'start_angle': run['phi_start'],
        'total_angle': run['phi_width'] * run['number_of_images'],
        'resolution': strategy['resolution'],
        'max_delta': run['phi_width'],
        'overlaps': run['overlaps'],
        'exposure_rate': -1,
    }
    if run.get('exposure_time', 0) > 0:
        info['exposure_rate'] = float(round(run['phi_width'], 1)) / round(run['exposure_time'], 1)
    return info


def screening_summary(analysis: Analysis, experiment: Experiment) -> dict:
    """
    Generate the summary table for the provided list of datasets
    :param analysis: list of dataset dictionaries
    :param experiment: Experiment to display
    :return: dictionary of table specification
    """

    indexing = analysis.get_step_result(experiment, StepType.INDEX)
    strategy = analysis.get_step_result(experiment, StepType.STRATEGY)

    if indexing is None or strategy is None:
        return {}
    else:
        lattice = indexing.get("lattice", Lattice())
        return {
            'title': 'Data Quality Statistics',
            'kind': 'table',
            'data': [
                ['Score¹', f'{strategy.get("quality.score"):0.2f}'],
                ['Wavelength', f'{experiment.wavelength:0.5g} Å'],
                ['Compatible Point Groups', ", ".join(indexing.get('point_groups'))],
                ['Reduced Cell²', f'{lattice.cell_text()}'],
                ['Mosaicity', f'{strategy.get("quality.mosaicity"):0.2f}°'],
                ['Diffraction Resolution³', f'{strategy.get("quality.resolution"):0.1f} Å'],
                ['Profile Error (position)', f'{indexing.get("quality.pixel_error")} px'],
                ['Profile Error (angle)', f'{indexing.get("quality.angle_error", -1)}°'],
                ['', ''],
                ['Expected Resolution', f'{strategy.get("strategy.resolution"):0.1f} Å'],
                ['Expected Completeness⁴', f'{strategy.get("strategy.completeness"):0.1f} %'],
                ['Expected Multiplicity⁴', f'{strategy.get("strategy.multiplicity"):0.2f}'],
            ],
            'header': 'column',
            'notes': inspect.cleandoc("""
                1. Data Quality Score for comparing similar data sets. Typically, values >
                   0.8 are excellent, > 0.6 are good, > 0.5 are acceptable, > 0.4
                   marginal, and &lt; 0.4 are Barely usable. Not comparable to full dataset scores.
                2. Strategy is calculated for the triclinic Reduced Cell which represents the worst case
                   scenario.
                3. This is the Resolution within which 99% of observed diffraction spots occur.
                4. This is the expected completeness and multiplicity for a triclinic crystal. If your crystal
                   has a higher symmetry, the observed completeness, and multiplicity for the final 
                   dataset will be much higher. 
            """)
        }


def screening_strategy(strategy):
    return {
        'title': 'Suggested Data Acquisition Strategy',
        'kind': 'table',
        'header': 'column',
        'data': [
            ['Recommended Detector Limit', f'{strategy.get("strategy.max_resolution"):0.2f} Å'],
            ['Attenuation', f'{strategy.get("strategy.attenuation"):0.2f}%'],
            ['Start Angle', f'{strategy.get("strategy.start_angle"):0.2f}°'],
            ['Maximum Delta Angle¹', f'{strategy.get("strategy.max_delta"):0.2f}°'],
            ['Minimum Total Angle Range²', f'{strategy.get("strategy.total_angle"):0.2f}°'],
            ['Exposure Rate Avg Dose³', (
                f'{strategy.get("strategy.exposure_rate"):0.1f}°/s | '
                f'0.2°/{0.2 / strategy.get("strategy.exposure_rate"):0.1f} s | '
                f'0.1°/{0.1 / strategy.get("strategy.exposure_rate"):0.1f} s '
            )],
            ['Exposure Rate Low Dose³', (
                f'{strategy.get("strategy.exposure_rate_worst"):0.1f}°/s | '
                f'0.2°/{0.2 / strategy.get("strategy.exposure_rate_worst"):0.1f} s | '
                f'0.1°/{0.1 / strategy.get("strategy.exposure_rate_worst"):0.1f} s '
            )],
            ['Total Exposure Time', f'{strategy.get("strategy.total_exposure"):0.0f} s'],
            ['Total Worst Case Exposure Time', f'{strategy.get("strategy.total_exposure_worst"):0.0f} s'],
        ],
        'notes': inspect.cleandoc("""
            1. This is the maximum delta-angle to be collected in order to avoid overlaps. Note that
               it may be desirable to use a smaller delta angle than this value to obtain better quality data, if the
               beamline allows.
            2. Minimum angle range for complete data. This is the bare minimum and it is strongly recommended to 
               to collect a full 180 degrees of data and often even more.
            3. Use the Avg Dose rate for Helical data collection. Use the Low Dose exposure rate if the crystal will
               not be translated during the experiment. Dose estimates make use of RADDOSE-3D: 
               see Bury et al PROTEIN SCIENCE 2018 VOL 27:217—228, https://doi.org/10.1002/pro.3302
             """),
    }


def screening_completeness(strategy):
    plots = defaultdict(list)
    for entry in strategy.get('statistics', []):
        total_range = int(entry['end_angle'] - entry['start_angle'])
        plots[total_range].append((entry['start_angle'], entry['completeness']))
    x_values = [x for x, y in next(iter(plots.values()))]

    return {
        'title': 'Minimal Range for Complete Data Acquisition',
        'kind': 'lineplot',
        'data': {
            'x': ['Start Angle'] + x_values,
            'y1': [
                [f'{total}°'] + [y for x, y in values]
                for total, values in plots.items()
            ],
            'y1-label': 'Completeness (%)'
        },
        'notes': "The above plot was calculated by XDS. See W. Kabsch, Acta Cryst. (2010). D66, 125-132."
    }


