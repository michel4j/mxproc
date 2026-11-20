import inspect
from collections import defaultdict

from .models import ExperimentReportData, StrategyReportData


def screening_summary(expt: ExperimentReportData) -> dict:
    """
    Generate the summary table for the provided screening experiment.
    """
    indexing = expt.indexing
    strategy = expt.strategy

    if indexing is None or strategy is None:
        return {}

    lattice = indexing.lattice
    sym_lattice = indexing.symmetry_lattice
    warnings = strategy.warnings
    warning_text = ""
    if warnings:
        warning_list = '\n'.join([f' - {warn}\n' for warn in warnings])
        warning_text = f"""
            ---
            **⚠️ WARNINGS ⚠️**
                
            {warning_list}
            """
    notes = inspect.cleandoc(
        f"""
            1. Data Quality Score for comparing similar data sets. Typically, values >
               0.8 are excellent, > 0.6 are good, > 0.5 are acceptable, > 0.4
               marginal, and &lt; 0.4 are Barely usable. Not comparable to full dataset scores.
            2. Strategy is calculated for the highest apparent symmetry which is unreliable for incomplete data.
            3. This is the Resolution within which 99% of observed diffraction spots occur.
            4. This is the expected completeness and multiplicity for a triclinic crystal. If your crystal
               has a higher symmetry, the observed completeness, and multiplicity for the final 
               dataset will be much higher.  
            
            {warning_text}
            """
    )

    strat = strategy.strategy
    return {
        'title': 'Data Quality Statistics',
        'kind': 'table',
        'data': [
            ['Score¹', f'{strategy.score:0.2f}'],
            ['Wavelength', f'{expt.wavelength:0.5g} Å'],
            ['Compatible Point Groups', ", ".join(indexing.point_groups)],
            ['Reduced Cell', f'{lattice.cell_text()}'],
            ['Apparent Point Group²', f'{sym_lattice.name}'],
            ['Apparent Cell²', f'{sym_lattice.cell_text()}'],
            ['Mosaicity', f'{strategy.mosaicity:0.2f}°'],
            ['Diffraction Resolution³', f'{strategy.resolution:0.1f} Å'],
            ['Profile Error (position)', f'{indexing.pixel_error} px'],
            ['Profile Error (angle)', f'{indexing.angle_error}°'],
            ['', ''],
            ['Expected Resolution', f'{strat["resolution"]:0.1f} Å'],
            ['Expected Completeness⁴', f'{strat["completeness"]:0.1f} %'],
            ['Expected Multiplicity⁴', f'{strat["multiplicity"]:0.2f}'],
        ],
        'header': 'column',
        'notes': notes,
    }


def screening_strategy(strategy: StrategyReportData) -> dict:
    strat = strategy.strategy
    return {
        'title': 'Suggested Data Acquisition Strategy',
        'kind': 'table',
        'header': 'column',
        'data': [
            ['Recommended Detector Limit', f'{strat["max_resolution"]:0.2f} Å'],
            ['Attenuation', f'{strat["attenuation"]:0.2f}%'],
            ['Start Angle', f'{strat["start_angle"]:0.2f}°'],
            ['Maximum Delta Angle¹', f'{strat["max_delta"]:0.2f}°'],
            ['Minimum Total Angle Range²', f'{strat["total_angle"]:0.2f}°'],
            ['Exposure Rate Avg Dose³', (
                f'{strat["exposure_rate"]:0.2g}°/s | '
                f'0.2°/{0.2 / strat["exposure_rate"]:0.2g} s | '
                f'0.1°/{0.1 / strat["exposure_rate"]:0.2g} s '
            )],
            ['Exposure Rate Low Dose³', (
                f'{strat["exposure_rate_worst"]:0.2g}°/s | '
                f'0.2°/{0.2 / strat["exposure_rate_worst"]:0.2g} s | '
                f'0.1°/{0.1 / strat["exposure_rate_worst"]:0.2g} s '
            )],
            ['Total Exposure Time', f'{strat["total_exposure"]:0.2g} s'],
            ['Total Worst Case Exposure Time', f'{strat["total_exposure_worst"]:0.2g} s'],
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


def screening_completeness(strategy: StrategyReportData) -> dict:
    plots = defaultdict(list)
    for entry in strategy.statistics:
        total_range = int(entry['end_angle'] - entry['start_angle'])
        plots[total_range].append((entry['start_angle'], entry['completeness']))
    x_values = [x for x, _ in next(iter(plots.values()))] if plots else []

    return {
        'title': 'Minimal Range for Complete Data Acquisition',
        'kind': 'lineplot',
        'data': {
            'x': ['Start Angle'] + x_values,
            'y1': [
                [f'{total}°'] + [y for _, y in values]
                for total, values in plots.items()
            ],
            'y1-label': 'Completeness (%)'
        },
        'notes': "The above plot was calculated by XDS. See W. Kabsch, Acta Cryst. (2010). D66, 125-132.  The plot"
                 " assumes the highest symmetry point group is the correct lattice. However, this may be false, as"
                 " a full dataset is required to accurately determine the correct symmetry."
    }
