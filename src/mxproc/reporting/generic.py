import inspect

from mxproc.xtal import Lattice, lattice_point_groups

from .models import (
    IntegrationReportData,
    QualitySummary,
    ReportPayload,
    ScalingReportData,
    SymmetryReportData,
    TwinningReportData,
)


def add_summary_column(
    report: dict,
    summary: QualitySummary,
    lattice: Lattice,
    name: str,
    wavelength: float,
):
    multiplicity = summary.observed / (summary.unique or 1)

    report['data'][0].append(name)
    report['data'][1].append(f'{summary.score:0.2f}')
    report['data'][2].append(f'{wavelength:0.5g} Å')
    report['data'][3].append(f'{lattice.name} ({lattice.spacegroup})')
    report['data'][4].append(f'{lattice.cell_text()}')
    report['data'][5].append(f'{summary.resolution:0.2f} Å')
    report['data'][6].append(f'{summary.observed}')
    report['data'][7].append(f'{summary.unique}')
    report['data'][8].append(f'{multiplicity:0.1f}')
    report['data'][9].append(f'{summary.completeness:0.1f} ({summary.anom_completeness:0.1f}) %')
    if summary.mosaicity is not None:
        report['data'][10].append(f'{summary.mosaicity:0.2f}')
    else:
        report['data'][10].append('N/A')

    report['data'][11].append(f'{summary.i_sigma:0.1f}')
    report['data'][12].append(f'{summary.r_meas:0.1f}')
    report['data'][13].append(f'{summary.cc_half:0.1f} %')
    if summary.i_sigma_a is not None:
        report['data'][14].append(f'{summary.i_sigma_a:0.1f}')
    else:
        report['data'][14].append('N/A')


def summary_table(payload: ReportPayload) -> dict:
    """
    Generate the summary table for the provided ReportPayload.
    """
    report = {
        'title': 'Data Quality Statistics',
        'kind': 'table',
        'data': [
            [''],
            ['Score¹'],
            ['Wavelength'],
            ['Space Group²⁶'],
            ['Unit Cell Parameters'],
            ['Resolution⁷'],
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
        'notes': (f"""
            1. Data Quality Score for comparing similar data sets. Typically, values >
               0.8 are excellent, > 0.6 are good, > 0.5 are acceptable, > 0.4
               marginal, and &lt; 0.4 are Barely usable
            2. POINTLESS was used for automatic spacegroup assignments (see P.R.Evans,
               Acta Cryst. D62, 72-82, 2005). The procedure is unreliable for incomplete datasets
               such as those used for screening. Please Inspect the detailed results below. Does
            3. Percentage correlation between intensities from random half-datasets. 
               (see Karplus & Diederichs (2012), Science. 336 (6084): 1030-1033)
            4. The highest I/Sigma(I) that the experimental setup can produce (Diederichs (2010) 
               Acta Cryst D66, 733-740).
            5. Anomalous completeness is shown in parentheses.
            6. Space group was {payload.symmetry_method}
            7. Resolution limit was set by {payload.resolution_method}
        """)
    }
    last_scaling = None
    wavelength = 0.0

    for experiment in payload.experiments:
        scaling = experiment.scaling
        symmetry = experiment.symmetry
        wavelength = experiment.wavelength

        if not scaling:
            continue
        last_scaling = scaling

        if payload.merge and symmetry is not None:
            add_summary_column(report, symmetry.summary, symmetry.lattice, experiment.name, wavelength)
        else:
            add_summary_column(report, scaling.summary, scaling.lattice, experiment.name, wavelength)

    if payload.merge and last_scaling is not None:
        add_summary_column(report, last_scaling.summary, last_scaling.lattice, 'COMBINED', wavelength)

    report['notes'] = inspect.cleandoc(report['notes'])
    return report


def lattice_table(lattices: list[dict]) -> dict:
    return {
        'title': "Lattice Character",
        'kind': 'table',
        'data': [
                    ['No.', 'Character', 'Error', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Point Groups', ]
                ] + [
                    [
                        lattice['index'],
                        lattice['character'],
                        f'{lattice["quality"]:0.1f}',
                        *(f'{val:0.1f}' for val in lattice['unit_cell']),
                        ', '.join(lattice_point_groups(lattice['character']))
                    ] for lattice in lattices
                ],
        'header': 'row',
        'notes': (
            "The Lattice Character is defined by the metrical parameters of its reduced cell as described "
            "in the International Tables for Crystallography Volume A, p. 746 (Kluwer Academic Publishers, "
            "Dordrecht/Boston/London, 1989). Note that more than one lattice character may have the "
            "same Bravais Lattice.  The error column indicates the quality of fit."
        ),
    }


def spacegroup_table(symmetry: SymmetryReportData) -> dict:
    return {
        'title': "Likely Space-Groups and their Probabilities",
        'kind': 'table',
        'data': [
                    ['Selected', 'Candidates', 'Space Group Number', 'Probability']
                ] + [
                    [
                        '*' if candidate['number'] == symmetry.lattice.spacegroup else '',
                        candidate['name'], candidate['number'], candidate['probability']
                    ] for candidate in symmetry.candidates
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


def standard_error_report(integration: IntegrationReportData) -> dict:
    errors = integration.errors[:-1]
    return {
        'title': 'Standard Errors of Reflection Intensities by Resolution',
        'content': [
            {
                'kind': 'lineplot',
                'style': 'half-height',
                'data': {
                    'x': ['Resolution Shell'] + [
                        round(min(row['resol_range']), 2) for row in errors
                    ],
                    'y1': [
                        ['Chi²'] + [row['chi_sq'] for row in errors]
                    ],
                    'y2': [
                        ['I/Sigma'] + [row['i_sigma'] for row in errors]
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
                            round(min(row['resol_range']), 2) for row in errors
                        ],
                        'y1': [
                            ['R-observed'] + [row['r_obs'] for row in errors],
                            ['R-expected'] + [row['r_exp'] for row in errors],
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


def shell_statistics_report(scaling: ScalingReportData) -> dict:
    stats = scaling.statistics
    return {
        'title': 'Statistics of Final Reflections by Shell',
        'content': [
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Resolution Shell'] + [round(row['shell'], 2) for row in stats],
                    'y1': [
                        ['Completeness (%)'] + [row['completeness'] for row in stats],
                        ['CC½'] + [row['cc_half'] for row in stats],
                    ],
                    'y2': [
                        ['R-meas'] + [row['r_meas'] for row in stats],
                    ],
                    'y2-label': 'R-factors (%)',
                    'x-scale': 'inv-square'
                }
            },
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Resolution Shell'] + [round(row['shell'], 2) for row in stats],
                    'y1': [
                        ['I/Sigma(I)'] + [row['i_sigma'] for row in stats],
                    ],
                    'y2': [
                        ['SigAno'] + [row['sig_ano'] for row in stats],
                    ],
                    'x-scale': 'inv-square'
                }

            },
            {
                'kind': 'table',
                'data': [
                            ['Shell', 'Observed', 'Unique', 'Completeness', 'R_obs', 'R_meas', 'CC½', 'I/Sigma(I)¹',
                             'SigAno²', 'CCₐₙₒ³']
                        ] + [
                            [
                                f'{row["shell"]:0.2f}',
                                f'{row["observed"]:d}',
                                f'{row["unique"]:d}',
                                f'{row["completeness"]:0.2f}',
                                f'{row["r_obs"]:0.2f}',
                                f'{row["r_meas"]:0.2f}',
                                f'{row["cc_half"]:0.2f}',
                                f'{row["i_sigma"]:0.2f}',
                                f'{row["sig_ano"]:0.2f}',
                                f'{row["cor_ano"]:0.2f}',
                            ] for row in stats
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


def frame_statistics_report(symmetry: SymmetryReportData) -> dict:
    frames = symmetry.frames
    diffs = symmetry.differences
    return {
        'title': 'Statistics of Intensities by Frame Number',
        'content': [
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in frames],
                    'y1': [
                        ['Intensity'] + [row['iobs'] for row in frames],
                    ],
                    'y2': [
                        ['Correlation'] + [row['corr'] for row in frames],
                    ],
                }
            },
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in frames],
                    'y1': [
                        ['R-meas'] + [row['r_meas'] for row in frames]
                    ],
                    'y2': [
                        ['I/Sigma(I)'] + [row['i_sigma'] for row in frames],
                    ]
                }
            },
            {
                'kind': 'scatterplot',
                'style': 'half-height',
                'data': {
                    'x': ['Frame Number'] + [row['frame'] for row in frames],
                    'y1': [
                        ['Reflections'] + [row['refs'] for row in frames]
                    ],
                    'y2': [
                        ['Unique'] + [row['unique'] for row in frames]
                    ],
                }
            },
            {
                'kind': 'lineplot',
                'data': {
                    'x': ['Frame Number Difference'] + [row['frame_diff'] for row in diffs],
                    'y1': [
                        ['All'] + [row['rd'] for row in diffs],
                        ['Friedel'] + [row['rd_friedel'] for row in diffs],
                        ['Non-Friedel'] + [row['rd_non_friedel'] for row in diffs],
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


def wilson_report(twinning: TwinningReportData) -> dict:
    wilson = twinning.plots
    return {
        'title': 'Wilson Statistics',
        'content':  [{
            'kind': 'lineplot',
            'title': 'Wilson Plot',
            'data': {
                'x': ['Resolution'] + [round((1 / row['inv_res_sq']) ** 0.5, 2) for row in wilson if row['mean_i'] > 0],
                'y1': [
                    ['Observed'] + [row['mean_i'] for row in wilson if row['mean_i'] > 0],
                    ['Expected'] + [row['expected_i'] for row in wilson if row['expected_i'] > 0],
                    ['Binned'] + [row['mean_i_binned'] for row in wilson if row['mean_i_binned'] > 0],
                ],
                'x-scale': 'inv-square',
                'y1-label': '⟨I⟩',
            },
            'notes': inspect.cleandoc("""
                *  This plot shows the falloff in intensity as a function in resolution; The expected
                   curve is based on analysis of structures in the PDB. Major deviations from the expected
                   plot may indicate pathological data or processing problems.
                *  The analysis was performed with XTRIAGE:  CCP4 newsletter No. 43, Winter 2005 from the PHENIX 
                   package:  Acta Cryst. D75, 861-877 (2019).
            """)
        }]
    }


def twinning_report(twinning: TwinningReportData) -> dict:
    if twinning.l_zscore:
        l_test = {
            'title': 'L-Test for twinning',
            'kind': 'lineplot',
            'data': {
                'x': ['|L|'] + [row['abs_l'] for row in twinning.l_test],
                'y1': [
                    ['Observed'] + [row['observed'] for row in twinning.l_test],
                    ['Twinned'] + [row['twinned'] for row in twinning.l_test],
                    ['Untwinned'] + [row['untwinned'] for row in twinning.l_test],
                ],
                'y1-label': 'P(L>=1)',
            },
            'notes': inspect.cleandoc("""
                *  <|L|>: {1:0.3f}  [untwinned: {2:0.3f}, perfect twin: {3:0.3f}]
                *  Multivariate Z-Score: {0:0.3f}.  The multivariate Z score is a quality measure of the 
                   given spread in intensities. Good to reasonable data are expected to have a Z score 
                   lower than 3.5.  Large values can indicate twinning, but small values 
                   do not necessarily exclude it.
                *  The analysis was performed with XTRIAGE:  CCP4 newsletter No. 43, Winter 2005 from the PHENIX 
                   package:  Acta Cryst. D75, 861-877 (2019).
            """.format(twinning.l_zscore, *twinning.l_statistic))
        }
    else:
        return {
            'title': 'Twinning Analysis',
            'description': 'Twinning analysis could not be performed.'
        }

    if twinning.laws:
        twin_laws = {
            'title': 'Twin Laws',
            'kind': 'table',
            'header': 'row',
            'data': [
                        ['Operator', 'Type', 'R', 'Britton alpha', 'H alpha', 'ML alpha'],
                    ] + [
                        [law['operator'], law['type'], law['r_obs'], law['britton_alpha'], law['H_alpha'],
                         law['ML_alpha']]
                        for law in twinning.laws
                    ],
            'notes': inspect.cleandoc("""
                *  Please note that the possibility of twin laws only means that the lattice symmetry
                   permits twinning; it does not mean that the data are actually twinned.
                   You should only treat the data as twinned if the intensity statistics are abnormal.
                *  The analysis was performed with XTRIAGE:  CCP4 newsletter No. 43, Winter 2005 from the PHENIX 
                   package:  Acta Cryst. D75, 861-877 (2019).
            """)
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
