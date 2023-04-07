import inspect

import numpy

from mxproc import Analysis, StepType, Experiment
from mxproc.xtal import Lattice


def summary_table(analysis: Analysis):
    """
    Generate the summary table for the provided list of datasets
    :param analysis: Analysis instance
    :return: dictionary of table specification
    """

    report = {
        'title': 'Data Quality Statistics',
        'kind': 'table',
        'data': [
            [''],
            ['Score¹'],
            ['Wavelength (A)'],
            ['Space Group²'],
            ['Unit Cell (A)'],
            ['Resolution⁶'],
            ['All Reflections'],
            ['Unique Reflections'],
            ['Multiplicity'],
            ['Completeness⁵'],
            ['Mosaicity'],
            ['I/Sigma(I)'],
            ['R-meas'],
            ['CC½³'],
            ['ISa⁴'],
        ],
        'header': 'column',
        'notes': ("""
            1. Data Quality Score for comparing similar data sets. Typically, values >
               0.8 are excellent, > 0.6 are good, > 0.5 are acceptable, > 0.4
               marginal, and &lt; 0.4 are Barely usable
            2. This space group was automatically assigned using POINTLESS (see P.R.Evans,
               Acta Cryst. D62, 72-82, 2005). This procedure is unreliable for incomplete datasets
               such as those used for screening. Please Inspect the detailed results below.
            3. Percentage correlation between intensities from random half-datasets. 
               (see Karplus & Diederichs (2012), Science. 336 (6084): 1030-1033)
            4. The highest I/Sigma(I) that the experimental setup can produce (Diederichs (2010) 
               Acta Cryst D66, 733-740).
            5. Anomalous completeness is shown in parentheses.
        """)
    }
    res_method = -1

    for experiment in analysis.experiments:
        result = analysis.get_step_result(experiment, StepType.SCALE)
        symmetry = analysis.get_step_result(experiment, StepType.SYMMETRY)
        if not result:
            continue

        multiplicity = result.get('quality.summary.observed', 0) / result.get('quality.summary.unique', 1)
        res_method = result.get('quality.summary.resolution_method', '')
        lattice = result.get("lattice", Lattice())
        mosaicity = result.get("quality.summary.mosaicity")
        completeness = result.get("quality.summary.completeness", 0)
        anom_completeness = result.get("quality.summary.anom_completeness", 0)

        i_sigma_a = symmetry.get("quality.summary.i_sigma_a")

        report['data'][0].append(experiment.name)
        report['data'][1].append(f'{result.get("quality.summary.score", 0.0):0.2f}')
        report['data'][2].append(f'{experiment.wavelength:0.5g}')
        report['data'][3].append(f'{lattice.name} ({lattice.spacegroup})')
        report['data'][4].append(f'{lattice.cell_text()}')
        report['data'][5].append(f'{result.get("quality.summary.resolution", 20.):0.2f}')
        report['data'][6].append(f'{result.get("quality.summary.observed", 0)}')
        report['data'][7].append(f'{result.get("quality.summary.unique", 0)}')
        report['data'][8].append(f'{multiplicity:0.1f}')
        report['data'][9].append(f'{completeness:0.1f} ({anom_completeness:0.1f}) %')
        if mosaicity is not None:
            report['data'][10].append(f'{mosaicity:0.2f}')
        else:
            report['data'][10].append('N/A')

        report['data'][11].append(f'{result.get("quality.summary.i_sigma", -99):0.1f}')
        report['data'][12].append(f'{result.get("quality.summary.r_meas", -99):0.1f}')
        report['data'][13].append(f'{result.get("quality.summary.cc_half", -99):0.1f} %')
        if i_sigma_a is not None:
            report['data'][14].append(f'{i_sigma_a:0.1f}')
        else:
            report['data'][14].append('N/A')

    report['notes'] += f"""    6. Resolution selection: {res_method}"""
    report['notes'] = inspect.cleandoc(report['notes'])

    return report


def lattice_table(analysis: Analysis, experiment: Experiment):
    indexing = analysis.get_step_result(experiment, StepType.INDEX)
    return {
        'title': "Lattice Character",
        'kind': 'table',
        'data': [
                    ['No.', 'Lattice type', 'Cell Parameters', 'Quality']
                ] + [
                    [
                        lattice['index'], lattice['character'],
                        '{:0.1f} {:0.1f} {:0.1f} {:0.1f} {:0.1f} {:0.1f}'.format(*lattice['unit_cell']),
                        '{:0.1f}'.format(lattice['quality']),
                    ] for lattice in indexing.get('lattices')
                ],
        'header': 'row',
        'notes': (
            "The Lattice Character is defined by the metrical parameters of its reduced cell as described "
            "in the International Tables for Crystallography Volume A, p. 746 (Kluwer Academic Publishers, "
            "Dordrecht/Boston/London, 1989). Note that more than one lattice character may have the "
            "same Bravais Lattice."
        ),
    }


def spacegroup_table(dataset, options):
    results = dataset['results']
    return {
        'title': "Likely Space-Groups and their Probabilities",
        'kind': 'table',
        'data': [
                    ['Selected', 'Candidates', 'Space Group Number', 'Probability']
                ] + [
                    [
                        '*' if sol['number'] == results['correction']['symmetry']['space_group'][
                            'sg_number'] else '',
                        xtal.SG_NAMES[sol['number']], sol['number'], sol['probability']
                    ] for sol in results['symmetry']['candidates']
                ],
        'header': 'row',
        'notes': (
            "The above table contains results from POINTLESS (see Evans, Acta Cryst. D62, 72-82, 2005). "
            "Indistinguishable space groups will have similar probabilities. If two or more of the top candidates "
            "have the same probability, the one with the fewest symmetry assumptions is chosen. "
            "This usually corresponds to the point group,  trying out higher symmetry space groups within the "
            "top tier does not require re-indexing the data as they are already in the same setting. "
            "For more detailed results, please inspect the output file 'pointless.log'."
        )
    }


def standard_error_report(dataset, options):
    results = dataset['results']
    return {
        'title': 'Standard Errors of Reflection Intensities by Resolution',
        'content': [
            {
                'kind': 'lineplot',
                'style': 'half-height',
                'data': {
                    'x': ['Resolution Shell'] + [
                        round(numpy.mean(row['resol_range']), 2) for row in
                        results['correction']['standard_errors'][:-1]
                    ],
                    'y1': [
                        ['Chi²'] + [row['chi_sq'] for row in results['correction']['standard_errors'][:-1]]
                    ],
                    'y2': [
                        ['I/Sigma'] + [row['i_sigma'] for row in results['correction']['standard_errors'][:-1]]
                    ],
                    'x-scale': 'inv-square'
                },

            },
            {
                'kind': 'lineplot',
                'style': 'half-height',
                'data':
                    {
                        'x': ['Resolution Shell'] + [
                            round(numpy.mean(row['resol_range']), 2) for row in
                            results['correction']['standard_errors'][:-1]
                        ],
                        'y1': [
                            ['R-observed'] + [row['r_obs'] for row in results['correction']['standard_errors'][:-1]],
                            ['R-expected'] + [row['r_exp'] for row in results['correction']['standard_errors'][:-1]],
                        ],
                        'y1-label': 'R-factors (%)',
                        'x-scale': 'inv-square'
                    }
            },
        ],
        'notes': inspect.cleandoc("""
            "* I/Sigma:  Mean intensity/Sigma of a reflection in shell"
            "* χ²: Goodness of fit between sample variances of symmetry-related intensities and their errors "
            "  (χ² = 1 for perfect agreement)."
            "* R-observed: Σ|I(h,i)-I(h)| / Σ[I(h,i)]"
            "* R-expected: Expected R-FACTOR derived from Sigma(I) """
                                  )
    }


def shell_statistics_report(dataset, options, ):
    results = dataset['results']
    analysis = results['correction'] if not 'scaling' in results else results['scaling']
    return {
        'title': 'Statistics of Final Reflections by Shell',
        'content': [
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Resolution Shell'] + [float(row['shell']) for row in analysis['statistics'][:-1]],
                    'y1': [
                        ['Completeness (%)'] + [row['completeness'] for row in analysis['statistics'][:-1]],
                        ['CC½'] + [row['cc_half'] for row in analysis['statistics'][:-1]],
                    ],
                    'y2': [
                        ['R-meas'] + [row['r_meas'] for row in analysis['statistics'][:-1]],
                    ],
                    'y2-label': 'R-factors (%)',
                    'x-scale': 'inv-square'
                }
            },
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Resolution Shell'] + [float(row['shell']) for row in analysis['statistics'][:-1]],
                    'y1': [
                        ['I/Sigma(I)'] + [row['i_sigma'] for row in analysis['statistics'][:-1]],
                    ],
                    'y2': [
                        ['SigAno'] + [row['sig_ano'] for row in analysis['statistics'][:-1]],
                    ],
                    'x-scale': 'inv-square'
                }

            },
            {
                'kind': 'table',
                'data': [
                            ['Shell', 'Completeness', 'R_meas', 'CC½', 'I/Sigma(I)¹', 'SigAno²', 'CCₐₙₒ³']
                        ] + [
                            [
                                row['shell'],
                                '{:0.2f}'.format(row['completeness']),
                                '{:0.2f}'.format(row['r_meas']),
                                '{:0.2f}'.format(row['cc_half']),
                                '{:0.2f}'.format(row['i_sigma']),
                                '{:0.2f}'.format(row['sig_ano']),
                                '{:0.2f}'.format(row['cor_ano']),
                            ] for row in analysis['statistics']
                        ],
                'header': 'row',
                'notes': inspect.cleandoc("""
                    1. Mean of intensity/Sigma(I) of unique reflections (after merging symmetry-related 
                       observations). Where Sigma(I) is the standard deviation of reflection intensity I 
                       estimated from sample statistics.
                    2. Mean anomalous difference in units of its estimated standard deviation 
                       (|F(+)-F(-)|/Sigma). F(+), F(-) are structure factor estimates obtained from the merged 
                       intensity observations in each parity class.
                    3. Percentage of correlation between random half-sets of anomalous intensity differences. """
                                          )
            }
        ]
    }


def frame_statistics_report(dataset, options):
    results = dataset['results']
    return {
        'title': 'Statistics of Intensities by Frame Number',
        'content': [
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in results['integration']['scale_factors']],
                    'y1': [
                        ['Scale Factor'] + [row['scale'] for row in results['integration']['scale_factors']],
                    ],
                    'y2': [
                        ['Mosaicity'] + [row['mosaicity'] for row in results['integration']['scale_factors']],
                        # ['Divergence'] + [row['divergence'] for row in results['integration']['scale_factors']],
                    ],
                }
            },
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in results['correction']['frame_statistics']],
                    'y1': [
                        ['R-meas'] + [row['r_meas'] for row in results['correction']['frame_statistics']]
                    ],
                    'y2': [
                        ['I/Sigma(I)'] + [row['i_sigma'] for row in results['correction']['frame_statistics']],
                    ]
                }
            },
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in results['integration']['scale_factors']],
                    'y1': [
                        ['Reflections'] + [row['ewald'] for row in results['integration']['scale_factors']]
                    ],
                    'y2': [
                        ['Unique'] + [row['unique'] for row in results['correction']['frame_statistics']]
                    ],
                }
            },
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Frame Number Difference'] + [row['frame_diff'] for row in
                                                        results['correction']['diff_statistics']],
                    'y1': [
                        ['All'] + [row['rd'] for row in results['correction']['diff_statistics']],
                        ['Friedel'] + [row['rd_friedel'] for row in results['correction']['diff_statistics']],
                        ['Non-Friedel'] + [row['rd_non_friedel'] for row in
                                           results['correction']['diff_statistics']],
                    ],
                    'y1-label': 'Rd'
                },
                'notes': inspect.cleandoc("""
                    *  The above plots use data generated by XDSSTAT. See Diederichs K. (2006) Acta Cryst D62, 96-101. 
                    *  Divergence: Estimated Standard Deviation of Beam divergence 
                    *  Rd: R-factors as a function of frame difference. An increase in R-d with frame difference is
                       suggestive of radiation damage."""
                                          )
            }
        ]
    }


def wilson_report(dataset, options):
    results = dataset['results']
    analysis = results['correction']
    if results.get('data_quality') and results['data_quality'].get('intensity_plots'):
        plot = {
            'kind': 'lineplot',
            'data': {
                'x': ['Resolution'] + [
                    (row['inv_res_sq']) ** -0.5 for row in results['data_quality']['intensity_plots']
                ],
                'y1': [
                    ['<I> Expected'] + [row['expected_i'] for row in results['data_quality']['intensity_plots']],
                ],
                'y2': [
                    ['<I> Observed'] + [row['mean_i'] for row in results['data_quality']['intensity_plots']],
                    ['<I> Binned'] + [row['mean_i_binned'] for row in results['data_quality']['intensity_plots']],
                ],

                'y1-label': '<I>',
                'x-scale': 'inv-square',
            }
        }
    else:
        plot = {
            'kind': 'lineplot',
            'data': {
                'x': ['Resolution'] + [
                    row['resolution'] for row in analysis['wilson_plot']
                ],
                'y1': [
                    ['<I> Observed'] + [row['mean_i'] for row in analysis['wilson_plot']],
                ],
                'x-scale': 'inv-square',
            },
            'notes': "Wilson fit: A={:0.3f}, B={:0.3f}, Correlation={:0.2f}".format(*analysis['wilson_line'])

        }
    return {
        'title': 'Wilson Statistics',
        'content': [
            plot
        ],
    }


def twinning_report(dataset, options):
    results = dataset['results']
    quality = results['data_quality']
    if quality.get('twinning_l_zscore'):
        l_test = {
            'title': 'L Test for twinning',
            'kind': 'lineplot',
            'data': {
                'x': ['|L|'] + [row['abs_l'] for row in quality['twinning_l_test']],
                'y1': [
                    ['Observed'] + [row['observed'] for row in quality['twinning_l_test']],
                    ['Twinned'] + [row['twinned'] for row in quality['twinning_l_test']],
                    ['Untwinned'] + [row['untwinned'] for row in quality['twinning_l_test']],
                ],
                'y1-label': 'P(L>=1)',
            },
            'notes': inspect.cleandoc("""
                *  <|L|>: {1:0.3f}  [untwinned: {2:0.3f}, perfect twin: {3:0.3f}]
                *  Multivariate Z-Score: {0:0.3f}.  The multivariate Z score is a quality measure of the 
                   given spread in intensities. Good to reasonable data are expected to have a Z score 
                   lower than 3.5.  Large values can indicate twinning, but small values 
                   do not necessarily exclude it""".format(quality['twinning_l_zscore'],
                                                           *quality['twinning_l_statistic'])
                                      )
        }
    else:
        return {
            'title': 'Twinning Analysis',
            'description': 'Twinning analysis could not be performed.'
        }

    if results['data_quality'].get('twin_laws'):
        laws = results['data_quality']['twin_laws']
        twin_laws = {
            'title': 'Twin Laws',
            'kind': 'table',
            'header': 'row',
            'data': [
                        ['Operator', 'Type', 'R', 'Britton alpha', 'H alpha', 'ML alpha'],
                    ] + [
                        [law['operator'], law['type'], law['r_obs'], law['britton_alpha'], law['H_alpha'],
                         law['ML_alpha']]
                        for law in laws
                    ],
            'notes': (
                "Please note that the possibility of twin laws only means that the lattice symmetry "
                "permits twinning; it does not mean that the data are actually twinned.  "
                "You should only treat the data as twinned if the intensity statistics are abnormal."
            )
        }
    else:
        twin_laws = {
            'title': 'Twin Laws',
            'description': 'No twin laws are possible for this crystal lattice.'
        }

    return {
        'title': 'Twinning Analysis',
        'content': [
            l_test,
            twin_laws
        ]
    }
