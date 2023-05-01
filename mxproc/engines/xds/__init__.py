from __future__ import annotations

import argparse
import os
import re
import shutil

from collections import defaultdict
from pathlib import Path
from typing import Sequence, Tuple
from mxio import XYPair

import numpy
import vg

from mxproc import Analysis, reporting
from mxproc.command import run_command, CommandFailed
from mxproc.common import StateType, StepType, backup_files, show_warnings, find_lattice, find_missing
from mxproc.common import generate_failure, Result, select_resolution, ScoreManager, summarize_ranges
from mxproc.engines.xds import io, stats
from mxproc.engines.xds.indexing import autoindex_trial
from mxproc.engines.xds.parser import XDSParser, IndexProblem, pointless
from mxproc.xtal import Experiment, Lattice, resolution_shells
from mxproc.programs import raddose
from mxproc.log import logger

MAX_INDEX_TRIES = 3
DEFAULT_MIN_SIGMA = 4


class IndexParamManager:
    """
    Stores and manages Indexing parameters across multiple trials, updating the parameters
    based on the problems detected during previous interation. Also keeps state so that degrees
    of suggested parameter changes would be based on the success or failure of previous attempts
    """

    degrees: dict  # parameter degrees
    params: dict  # previous parameters
    min_sigma: float  # Maximum Spot Sigma Value
    max_sigma: float  # Minimum Spot Sigma Value

    def __init__(
            self,
            data_range: Tuple[int, int],
            spot_range: Sequence[Tuple[int, int]],
            skip_range: Sequence[Tuple[int, int]],
            min_sigma: float = DEFAULT_MIN_SIGMA,
            max_sigma: float = 100,
            **extras
    ):
        self.degrees = {}
        self.params = {'spot_range': spot_range, "data_range": data_range, 'skip_range': skip_range, **extras}
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def get_degree(self, name: str) -> int:
        """
        Update the degree and return its count
        :param name: name of parameter
        :return: integer count
        """

        self.degrees[name] = self.degrees.get(name, 0) + 1
        return self.degrees[name]

    def get_parameters(self) -> dict:
        """
        Return the current parameter dictionary
        """
        return self.params

    def update_parameters(
            self,
            flags: IndexProblem,
            spot_range: Sequence[Tuple[int, int]] | None = None
    ) -> Tuple[bool, Sequence[str]]:
        """
        Determine modified indexing parameters based on Indexing problem flags
        :param flags: integer representing bit-wise combination of flags
        :param spot_range: Optional spot range parameter
        :return: indexing parameters, bool True if parameters have changed, sequence of message strings
        """

        messages = []
        request_retry = False

        if IndexProblem.INVERTED_AXIS in flags:
            self.params['invert_spindle'] = True
            messages.append('Inverting the rotation direction')
            if self.get_degree('invert_spindle') == 1:
                request_retry = True

        if IndexProblem.WRONG_SPOT_PARAMETERS in flags:
            self.params['spot_size'] = 0.5 / self.get_degree('spot_size')
            messages.append("Reducing accepted spot size & separation")
            request_retry = True

        if (IndexProblem.LOW_INDEXED_FRACTION | IndexProblem.POOR_SOLUTION) in flags and spot_range:
            self.params['spot_range'] = [spot_range]
            messages.append("Changing spot range")
            request_retry = True
        elif IndexProblem.POOR_SOLUTION in flags:
            self.params['error_scale'] = 2.0 * self.get_degree('error_scale')
            self.params['spot_separation'] = 1.0 - 0.25 ** self.get_degree('spot_separation')
            self.params['spot_size'] = 1.0 + 0.5 * self.get_degree('spot_size')
            messages.append("Relaxing quality criteria")
            request_retry = True

        if IndexProblem.REFINEMENT_FAILED in flags:
            refinement_flags = ('CELL', 'BEAM', 'ORIENTATION', 'AXIS',)[:-self.get_degree('refine_index')]
            if len(refinement_flags) > 1:
                self.params['refine_index'] = refinement_flags
                messages.append("Adjusting refinement parameters")
                request_retry = True

        if IndexProblem.INSUFFICIENT_SPOTS in flags:
            self.min_sigma = DEFAULT_MIN_SIGMA - self.get_degree('min_sigma')
            request_retry = True

        return request_retry, messages


class XDSAnalysis(Analysis):
    prefix = 'xds'

    def get_extras(self, args: argparse.Namespace) -> dict:
        extras = {}
        if args.frames:
            extras.update(
                data_range=(args.frames[0][0], args.frames[-1][1]),
                skip_range=find_missing(args.frames),
                spot_range=args.frames,
            )

        if args.beam_center:
            extras.update(beam_center=XYPair(*args.beam_center))

        if args.anom is not None:
            extras.update(anomalous=args.anom)

        if args.min_sigma:
            extras.update(min_sigma=args.min_sigma)

        if args.max_sigma:
            extras.update(max_sigma=args.max_sigma)

        if args.resolution:
            extras.update(resolution_limit=args.resolution)

        if args.spacegroup:
            extras.update(lattice=Lattice(spacegroup=args.spacegroup))

        return extras

    def initialize(self, **kwargs):
        results = {}
        logger.info('Working Directories:')
        for experiment in self.experiments:
            # create sub-directories if multiple processing
            directory = self.options.directory
            directory = directory if len(self.experiments) == 1 else directory / experiment.name
            directory.mkdir(parents=True, exist_ok=True)
            self.options.working_directories[experiment.identifier] = directory
            os.chdir(self.options.working_directories[experiment.identifier])
            io_options = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]), 'spot_range': experiment.frames,
                'skip_range': experiment.missing, 'beam_center': experiment.detector_origin
            }
            io_options.update(self.options.extras)
            io.create_input_file(('ALL',), experiment, io.XDSParameters(**io_options))
            path_str = str(directory)
            path_str = path_str.replace(os.path.expanduser('~'), '~', 1)
            logger.info_value(f'{experiment.name}', path_str)
            results[experiment.identifier] = Result(state=StateType.SUCCESS, details={"directory": directory})
        return results

    def find_spots(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            io_options = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]), 'spot_range': experiment.frames,
                'skip_range': experiment.missing, 'beam_center': experiment.detector_origin
            }
            io_options.update(self.options.extras)
            os.chdir(self.options.working_directories[experiment.identifier])
            io.create_input_file(('XYCORR', 'INIT', 'COLSPOT'), experiment, io.XDSParameters(**io_options))
            image_range = summarize_ranges(io_options['spot_range'])

            try:
                run_command('xds_par', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
                result = Result(state=StateType.SUCCESS, details=XDSParser.parse('COLSPOT.LP'))
                io.save_spots()
                backup_files('SPOT.XDS')
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            results[experiment.identifier] = result
        return results

    def index(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            # skip if the find_spots step did not succeed
            if not self.get_step_result(experiment, StepType.SPOTS):
                continue

            os.chdir(self.options.working_directories[experiment.identifier])

            result = generate_failure('')
            logger.info(f'{experiment.name}:')
            io_options = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]),
                'spot_range': experiment.frames, 'skip_range': experiment.missing,
                'anomalous': self.options.anomalous, 'beam_center': experiment.detector_origin
            }
            io_options.update(**self.options.extras)

            param_manager = IndexParamManager(**io_options)
            for trial_number in range(MAX_INDEX_TRIES):
                backup_files('IDXREF.LP')
                result, retry_requested, retry_messages = autoindex_trial(experiment, param_manager, trial_number)
                if retry_requested:
                    show_warnings(f'Retrying Indexing:', retry_messages)
                else:
                    break

            # show final unit cell parameters
            if result.get('point_groups'):
                point_groups = ", ".join(result.get('point_groups'))
                lattice = result.get('lattice', Lattice())
                logger.info_value('- Reduced Cell', lattice.cell_text())
                logger.info_value('- Compatible Point Groups', point_groups)
                logger.info_value('- Unit-cell Volume', f"{lattice.volume():0.1f} Å³")
                logger.info_value('- Estimated Unit Cell Content', f"{lattice.estimate_content()} Residues")

            results[experiment.identifier] = result
            if result.state in [StateType.WARNING, StateType.FAILURE]:
                show_warnings(f"{result.state.name}: Possible Indexing Problems", result.messages)

        return results

    def strategy(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])
            index_result = self.get_step_result(experiment, StepType.INDEX)
            lattice = index_result.get('lattice')

            io_options = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]),
                'spot_range': experiment.frames, 'skip_range': experiment.missing,
                'anomalous': self.options.anomalous, 'lattice': lattice,
                'beam_center': experiment.detector_origin,
                'min_fraction': 0.10
            }
            io_options.update(**self.options.extras)
            io.filter_spots()
            io.create_input_file(('IDXREF', 'DEFPIX', 'XPLAN'), experiment, io.XDSParameters(**io_options))

            try:
                run_command(
                    'xds_par', desc=f'{experiment.name}: Calculating optimal strategy', check_files=('XPLAN.LP',)
                )
                details = XDSParser.parse('XPLAN.LP')
            except (FileNotFoundError, CommandFailed) as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                cell_axis = {
                    ("a", lattice.a): numpy.array(index_result.get('cell_a_axis')),
                    ("b", lattice.b): numpy.array(index_result.get('cell_b_axis')),
                    ("c", lattice.c): numpy.array(index_result.get('cell_c_axis')),
                }

                beam_axis = numpy.array(index_result.get('beam_axis'))
                spindle_axis = numpy.array(index_result.get('rotation_axis'))
                axis_angles = [
                    {
                        'name': name,
                        'beam': vg.angle(beam_axis, axis),
                        'spindle': vg.angle(spindle_axis, axis),
                        'axis': axis,
                        'length': length
                    }
                    for (name, length), axis in cell_axis.items()
                ]
                axis_near_beam = min(axis_angles, key=lambda x: x['beam'])
                axis_near_spindle = min(axis_angles, key=lambda x: x['spindle'])
                longest_axis = max(axis_angles, key=lambda x: x['length'])
                axis_lengths = numpy.array([lattice.a, lattice.b, lattice.c])
                axis_ratios = numpy.round(axis_lengths / longest_axis['length'], 1)

                has_long_axis = ((axis_ratios <= 0.5).sum() == 2)  # two other axis are less than 50% length of longest
                long_axis_ratio = 0.0

                if longest_axis != axis_near_spindle and has_long_axis:
                    logger.warning_value(
                        "- Long Unit-cell axis is closest to beam axis!",
                        f'{longest_axis["name"]}={longest_axis["length"]:0.1f}Å, {longest_axis["beam"]:0.1f}°'
                    )
                    long_axis_ratio = longest_axis['beam'] / max(longest_axis['spindle'], 1)
                else:
                    logger.info_value(
                        f'- Unit-Cell axis closest to beam',
                        f'{axis_near_beam["name"]}, {axis_near_beam["beam"]:0.1f}°'
                    )
                resolution_details = stats.determine_resolution(experiment)

                # calculate maximum delta to avoid overlaps - see Dauter, Acta Cryst. (1999). D55, p1707
                mosaicity = index_result.get('quality.mosaicity', 0.1)
                max_delta = numpy.degrees(resolution_details['desired_resolution'] / axis_near_beam['length']) - mosaicity

                best_strategy = details['strategies'][-1]
                score_manager = ScoreManager({
                    'completeness':  (75.0, 0.2, 0.25),
                    'resolution': (2.0, 0.2, -3),
                    'long_axis_ratio': (0, 0.1, -1),
                    'mosaicity': (0.3, 0.1, -30),
                    'angle_error': (0.5, 0.1, -2),
                    'pixel_error': (2.0, 0.1, -2),
                    'indexed_fraction': (0.5, 0.2, 6),
                })

                # scoring
                score = score_manager.score(
                    completeness=best_strategy['completeness'],
                    resolution=resolution_details['observed_resolution'],
                    long_axis_ratio=long_axis_ratio,
                    mosaicity=mosaicity,
                    angle_error=index_result.get('quality.angle_error'),
                    pixel_error=index_result.get('quality.pixel_error'),
                    indexed_fraction=resolution_details['indexed_fraction'],
                )

                # run raddose and estiamte exposure time, assumes crystal is the same size as the beam
                # uses beam shape and flux from beamline passed in as options.
                beam = self.options.get_beam()
                beam.wavelength = experiment.wavelength
                success, dose_info = raddose.get_dose(
                    crystal_size=beam.aperture, total_range=best_strategy['total_angle'], lattice=lattice, beam=beam
                )

                details.update({
                    'strategy': {
                        **best_strategy,
                        'resolution': resolution_details['desired_resolution'],
                        'max_resolution': resolution_details['maximum_resolution'],
                        'max_delta': max_delta,
                        'attenuation': 0.0,
                        'exposure_rate': dose_info.get('exposure_rate', -1),
                        'exposure_rate_worst': dose_info.get('exposure_rate_worst', -1),
                        'total_exposure': dose_info.get('total_exposure', -1),
                        'total_exposure_worst': dose_info.get('total_exposure_worst', -1),
                    },
                    'quality': {
                        'mosaicity': mosaicity,
                        'resolution': resolution_details['observed_resolution'],
                        'long_axis': long_axis_ratio,
                        'score': score,
                    },
                    'dose': dose_info
                })
                result = Result(state=StateType.SUCCESS, details=details)
                self.show_strategy(result)
            results[experiment.identifier] = result
        return results

    def integrate(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])

            if not self.get_step_result(experiment, StepType.INDEX):
                continue

            logger.info(f'{experiment.name}:')
            io_options = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]),
                'spot_range': experiment.frames, 'skip_range': experiment.missing,
                'anomalous': self.options.anomalous,
            }
            io_options.update(**self.options.extras)
            #FIXME:  when overridden, update range_text
            range_text = summarize_ranges(experiment.frames)

            previous_integration = self.get_step_result(experiment, StepType.INTEGRATE)
            if self.options.optimize and previous_integration is not None:
                # copy parameter to new file
                shutil.copy('GXPARM.XDS', 'XPARM.XDS')

                # inject parameters to extra options
                io_options.update(
                    refl_range=previous_integration.get('parameters.refl_range'),
                    refl_range_esd=previous_integration.get('parameters.refl_range_esd'),
                    divergence=previous_integration.get('parameters.divergence'),
                    divergence_esd=previous_integration.get('parameters.divergence_esd'),
                )

            io.create_input_file(('DEFPIX', 'INTEGRATE', 'CORRECT',), experiment, io.XDSParameters(**io_options))
            try:
                run_command(
                    'xds_par', desc=f'- Integrating frames {range_text}', check_files=['INTEGRATE.LP', 'CORRECT.LP']
                )
                integration = XDSParser.parse('INTEGRATE.LP')
                correction = XDSParser.parse('CORRECT.LP')
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                details = {
                    'parameters': integration['parameters'],
                    'lattices': correction['lattices'],
                    'quality': {
                        'batches': integration['batches'],
                        'frames': integration['frames'],
                        'statistics': correction['statistics'],
                        'summary': correction['summary'],
                        'errors': correction['errors'],
                    },
                }
                result = Result(state=StateType.SUCCESS, details=details)
            results[experiment.identifier] = result

        return results

    def symmetry(self, **kwargs):
        results = {}
        reference = self.experiments[0]
        symmetry_experiments = []

        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])

            if self.get_step_result(experiment, StepType.INTEGRATE):
                logger.info(f'{experiment.name}:')
                symmetry_experiments.append(experiment)
                try:
                    run_command(
                        'pointless xdsin INTEGRATE.HKL xmlout pointless.xml', desc=f'- Determining symmetry',
                        check_files=['pointless.xml']
                    )
                    details = pointless.parse_pointless('pointless.xml')
                    experiment.lattice = details['lattice']
                except CommandFailed as err:
                    result = generate_failure(f"Command failed: {err}")
                else:
                    result = Result(state=StateType.SUCCESS, details=details)
                    lattice_message = f'{experiment.lattice.name} - #{experiment.lattice.spacegroup}'
                    logger.info_value(f'- Recommended {details["type"]}', lattice_message)
                    if experiment.lattice.spacegroup > reference.lattice.spacegroup:
                        reference = experiment
                results[experiment.identifier] = result

        for experiment in symmetry_experiments:
            if experiment.identifier not in results:
                continue
            os.chdir(self.options.working_directories[experiment.identifier])

            result = self.apply_symmetry(experiment, reference)
            result.details['symmetry'] = results[experiment.identifier].details
            results[experiment.identifier] = result

            if result.state == StateType.SUCCESS:
                logger.info_value(f'- Refined Cell:', result.get('lattice').cell_text())
                self.show_quality(result.details['quality'])

        return results

    def apply_symmetry(self, experiment: Experiment, reference: Experiment) -> Result:
        """
        Apply specified symmetry to the experiment and reindex the data
        :param experiment: Experiment instance
        :param reference: Reference experiment
        """
        logger.info(f'{experiment.name}:')
        integrate_result = self.get_step_result(experiment, StepType.INTEGRATE)
        target_lattice = self.options.extras.get('lattice', reference.lattice)
        if target_lattice != reference.lattice:
            self.settings['symmetry_method'] = 'manually chosen'
        else:
            self.settings['symmetry_method'] = 'automatically assigned'

        reindex_lattice, reindex_matrix = find_lattice(target_lattice, integrate_result.details['lattices'])
        if reindex_lattice:
            if reference != experiment:
                directory = self.options.working_directories[experiment.identifier]
                reference_directory = self.options.working_directories[reference.identifier]
                reference_data = Path(os.path.relpath((reference_directory / "XDS_ASCII.HKL"), directory))
            else:
                reference_data = None

            try:
                io_options = {
                    "data_range": (experiment.frames[0][0], experiment.frames[-1][1]),
                    "spot_range": experiment.frames, "skip_range": experiment.missing,  "reference": reference_data,
                }
                io_options.update(**self.options.extras)
                io_options.update(lattice=reindex_lattice, reindex=reindex_matrix)
                io.create_input_file(('CORRECT',), experiment, io.XDSParameters(**io_options))
                run_command(
                    'xds_par', desc=f'- Applying symmetry {reindex_lattice.name} - #{reindex_lattice.spacegroup}'
                )
                run_command('echo "XDS_ASCII.HKL" | xdsstat 20 3 > XDSSTAT.LP', desc=f'- Gathering extra statistics')
                correction = XDSParser.parse('CORRECT.LP')
                parameters = XDSParser.parse('GXPARM.XDS')
                stats_tables = XDSParser.parse('XDSSTAT.LP')
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                resolution, resolution_method = select_resolution(
                    correction['statistics'], manual=self.options.extras.get('resolution_limit')
                )
                self.settings['resolution_method'] = resolution_method
                details = {
                    'parameters': parameters,
                    'lattice': Lattice(parameters['sg_number'], *parameters['unit_cell']),
                    'lattices': correction['lattices'],
                    'quality': {
                        'frames': stats_tables['frame_statistics'],
                        'differences': stats_tables['diff_statistics'],
                        'statistics': correction['statistics'],
                        'summary': {
                            **correction['quality'], **correction['summary'],
                            "i_sigma_a": correction['correction']['parameters']['i_sigma_a'],
                            "resolution": resolution,
                            "resolution_method": resolution_method
                        },
                        'inner_shell': correction['statistics'][0],
                        'outer_shell': correction['statistics'][-1],
                        'errors': correction['errors'],
                    },
                    'wilson': correction['wilson'],
                    'correction': correction['correction'],
                }
                details['quality']['summary']['score'] = self.score(details['quality'])
                result = Result(state=StateType.SUCCESS, details=details)
        else:
            logger.warning(f'{experiment.name} not compatible with selected spacegroup #{target_lattice.spacegroup}')
            result = generate_failure(f"Incompatible Lattice type: {target_lattice.character}")
        return result

    def scale(self, **kwargs):
        scalable_experiments = {
            experiment.name: experiment
            for experiment in self.experiments
            if self.get_step_result(experiment, StepType.SYMMETRY)
        }
        if not scalable_experiments:
            logger.error('Skipping! No datasets available to scale!')
            return {}

        anom_option = self.options.anomalous
        resolution_limit = self.options.extras.get('resolution_limit', 0.0)

        scale_configs = []
        quality_contrib = defaultdict(list)
        self.options.merge &= len(scalable_experiments) > 1
        if self.options.merge:
            method = "Combined Output"
            # prepare merge parameters
            inputs = []
            resolutions = []
            for name, experiment in scalable_experiments.items():
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, StepType.SYMMETRY)
                quality_contrib["XSCALE.HKL"].append(symmetry_result.get('quality.summary'))
                input_file = str(Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory)))
                resolution = symmetry_result.get('quality.summary.resolution')
                inputs.append({'input_file': input_file, 'resolution': resolution})
                resolutions.append(resolution)

            # one configuration for merging
            scale_configs.append({
                'anomalous': anom_option,
                'shells': resolution_shells(max(resolution_limit, min(resolutions))),
                'output_file': "XSCALE.HKL",
                'inputs': inputs,
            })
            backup_files('XSCALE.HKL', 'XSCALE.LP')
        else:
            method = "Separate Output"
            for name, experiment in scalable_experiments.items():
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, StepType.SYMMETRY)
                input_file = str(Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory)))
                output_file = str(Path(os.path.relpath(directory / "XSCALE.HKL", self.options.directory)))
                resolution = symmetry_result.get('quality.summary.resolution')
                quality_contrib[output_file].append(symmetry_result.get('quality.summary'))
                scale_configs.append({
                    'anomalous': anom_option,
                    'shells': resolution_shells(max(resolution, resolution_limit)),
                    'output_file': output_file,
                    'inputs': [{'input_file': input_file, 'resolution': resolution}],
                })

        try:
            io.write_xscale_input(scale_configs)
            run_command('xscale_par', f'- Scaling {len(scalable_experiments)} dataset(s) for {method}')
            scaling = XDSParser.parse_xscale()
        except CommandFailed as err:
            result = generate_failure(f'Command Failed: {err}')
            results = {
                experiment.identifier: result
                for experiment in scalable_experiments.values()
            }
        else:
            # check twinning and prepare details
            details = {}
            for key, section in scaling.items():

                resolution, resolution_method = select_resolution(
                    section['statistics'], manual=self.options.extras.get('resolution_limit')
                )
                self.settings['resolution_method'] = resolution_method
                quality = self.assess_quality(section['output_file'])

                # xtriage completeness is a fraction
                if quality.get('summary', {}):
                    quality['summary']['completeness'] *= 100
                    quality['summary']['anom_completeness'] *= 100

                # carry forward some quality parameters from symmetry step
                # as averages of contributing datasets
                extra_stats = {
                    k: numpy.mean([d.get(k, 0.0) for d in quality_contrib[key]])
                    for k in ('pixel_error', 'angle_error', 'mosaicity', 'i_sigma_a')
                }
                details[key] = {
                    'output': section['output_file'],
                    'lattice': Lattice(section['spacegroup'], *section['unit_cell']),
                    'quality': {
                        'statistics': section['statistics'],
                        'summary': {
                            **section['summary'],
                            **quality.get('summary', {}),
                            **extra_stats,
                            'resolution': resolution,
                            'resolution_method': resolution_method,
                        },
                        'inner_shell': section['statistics'][0],
                        'outer_shell': section['statistics'][-1],
                        'twinning': quality.get('twinning', {}),
                    },
                    'wilson': section['wilson'],
                    'datasets': section['datasets'],
                    'correlations': section['correlations'],
                }
                details[key]['quality']['summary']['score'] = self.score(details[key]['quality'])
                logger.info_value('Final Data Quality', key)
                self.show_quality(details[key]['quality'])

            # for merging, all experiments get the single section
            exportable = defaultdict(list)
            results = {}
            for name, experiment in scalable_experiments.items():
                work_dir = self.options.working_directories[experiment.identifier]
                if self.options.merge:
                    scale_file = 'XSCALE.HKL'
                else:
                    scale_file = (work_dir / 'XSCALE.HKL').relative_to(self.options.directory)
                expt_details = details[str(scale_file)]
                results[experiment.identifier] = Result(state=StateType.SUCCESS, details=expt_details)
                exportable[expt_details['output']].append(name)

            self.settings['outputs'] = dict(exportable)

        return results

    @staticmethod
    def assess_quality(data_file: str | Path) -> dict:
        """
        Check the data quality for the specified file and return a dictionary of results
        :param data_file: reflection file
        """

        try:
            log_file = str(Path(data_file).absolute().parent / "XTRIAGE.LP")
            command = f'phenix.xtriage {data_file} loggraphs=True nproc=8 log={log_file}'
            run_command(command, f'- Assessing quality of "{data_file}"', check_files=[log_file])
            quality = XDSParser.parse(log_file)
        except CommandFailed:
            quality = {}

        return quality

    def export(self, **kwargs):
        for scaled_input, names in self.settings.get('outputs', {}).items():
            # get common name and strip special characters from ending.
            prefix = re.sub("[-_+]$", '', os.path.commonprefix(names)) or "final"

            # SHELX
            io.write_xdsconv_input({
                'format': 'SHELX',
                'anomalous': self.options.anomalous,
                'input_file': scaled_input,
                'prefix': prefix,
            })
            run_command('xdsconv', f'- Exporting SHELX reflection file "{prefix}-shelx.hkl"')

            # MTZ
            command = f'phenix.reflection_file_converter {scaled_input} --generate_r_free_flags  --mtz={prefix}.mtz'
            run_command(command, f'- Exporting MTZ reflection file "{prefix}.mtz"')

    @staticmethod
    def show_quality(info: dict):
        """
        Display the data quality details
        :param info: Quality dictionary
        """
        quality = Result(state=StateType.SUCCESS, details=info)

        i_sigma_a = quality.get('summary.i_sigma_a')
        completeness = quality.get('summary.completeness')
        resolution = quality.get('summary.resolution')
        lo_r_meas = quality.get('inner_shell.r_meas')
        lo_i_sigma = quality.get('inner_shell.i_sigma')
        resolution_method = quality.get('summary.resolution_method')
        score = quality.get('summary.score')

        logger.info_value(f'- Inner-Shell R-meas:', f'{lo_r_meas:0.1f} %')
        logger.info_value(f'- Inner-Shell I/Sigma:', f'{lo_i_sigma:0.1f}')
        logger.info_value(f'- Completeness:', f'{completeness:0.1f} %')
        logger.info_value(f'- Resolution limit (by {resolution_method}):', f'{resolution:0.2f}')
        if i_sigma_a is not None:
            logger.info_value(f'- I/Sigma(I) Asymptote [ISa]:', f'{i_sigma_a:0.1f}')

        logger.info_value(f'- Data Quality Score', f"{score:0.2}")

    @staticmethod
    def show_strategy(info: Result):
        """
        Display the screening strategy results
        :param info: strategy
        """
        start_angle = info.get('strategy.start_angle', 0)
        angle_range = info.get('strategy.total_angle', 180.0)
        completeness = info.get('strategy.completeness')
        max_delta = info.get('strategy.max_delta')
        multiplicity = info.get('strategy.multiplicity')
        mosaicity = info.get('quality.mosaicity')
        observed_resolution = info.get('quality.resolution')
        desired_resolution = info.get('strategy.resolution')
        exposure_rate = info.get('strategy.exposure_rate')
        exposure_rate_worst = info.get('strategy.exposure_rate_worst')
        total_exposure = info.get('strategy.total_exposure')
        total_exposure_worst = info.get('strategy.total_exposure_worst')
        score = info.get('quality.score')

        logger.info_value(f'- Starting angle', f'{start_angle:0.1f}°')
        logger.info_value(f'- Total Oscillation Range', f'{angle_range:0.1f}°')
        logger.info_value(f'- Predicted Completeness:', f'{completeness:0.1f} %')
        logger.info_value(f'- Predicted Multiplicity:', f'{multiplicity:0.1f}')
        logger.info_value(f'- Crystal Mosaicity:', f'{mosaicity:0.2f}')
        logger.info_value(f'- Maximum Delta to avoid overlaps', f'{max_delta:0.2f}°')
        logger.info_value(f'- Resolution of Diffraction (99-th percentile)', f'{observed_resolution:0.2f} Å')
        logger.info_value(f'- Recommended High-Resolution Limit', f'{desired_resolution:0.2f} Å')
        logger.info_value(
            f'- Maximum Exposure Rate (recommended, worst-case)',
            f'{exposure_rate:0.1f}°/s, {exposure_rate_worst:0.1f}°/s'
        )
        logger.info_value(
            f'- Maximum Total Exposure (recommended, worst-case)',
            f'{total_exposure:0.0f} s, {total_exposure_worst:0.0f} s'
        )
        logger.info_value(f'- Screening Score', f"{score:0.2f}")

    @staticmethod
    def score(info: dict) -> float:
        """
        Calculate and return a data quality score for the quality dictionary

        :param info:
        :return: float
        """

        manager = ScoreManager({
            'resolution': (2.0, 0.2, -6),
            'completeness': (75, 0.2, 0.25),
            'r_meas': (6, 0.2, -1),
            'i_sigma': (30, 0.1, 0.1),
            'mosaicity': (0.3, 0.1, -30),
            'angle_error': (1.0, 0.05, -2),
            'pixel_error': (2.0, 0.05, -2),
            'misfit_fraction': (0.1, 0.1, -6),
        })

        score = 0.0

        if info:
            # used Result object for easy access to deep fields with defaults
            quality = Result(state=StateType.SUCCESS, details=info)

            resolution = quality.get('summary.resolution', 20.0)
            completeness = quality.get('summary.completeness', 0.0)
            r_meas = quality.get('inner_shell.r_meas', 100.0)
            i_sigma = quality.get('inner_shell.i_sigma', 0.0)
            mosaicity = quality.get('summary.mosaicity', 0.0)
            angle_error = quality.get('summary.angle_error', 0.0)
            pixel_error = quality.get('summary.pixel_error', 0.0)
            misfit_fraction = quality.get('summary.misfits', 0.0) / quality.get('summary.reflections', 1.0)

            score = manager.score(
                resolution=resolution,
                completeness=completeness,
                r_meas=r_meas,
                i_sigma=i_sigma,
                mosaicity=mosaicity,
                angle_error=angle_error,
                pixel_error=pixel_error,
                misfit_fraction=misfit_fraction
            )
        return score

    def report(self, **kwargs):
        anom = 'Native' if not self.options.anomalous else 'Anomalous'
        if self.options.screen:
            names = []
            scores = []
            strategies = []
            for expt in self.experiments:
                strategy = self.get_step_result(expt, StepType.STRATEGY)
                if strategy:
                    strategies.append(strategy.get('strategy'))
                    names.append(expt.name)
                    scores.append(strategy.get('quality.score'))

            if scores and strategies:
                name_label = ", ".join(names)
                report = {
                    'title': f"{anom} Screening of {name_label}",
                    'kind': 'MX Screening',
                    'score': round(scores[0], 2),
                    'strategy': strategies[0],
                    'details': reporting.screening_report(self),
                }
                report_file = str(self.options.directory / "report.html").replace(os.path.expanduser('~'), '~', 1)
                logger.info(f'- HTML report: {report_file}')
                reporting.save_report(report, self.options.directory)
            else:
                raise CommandFailed('No successful results to report on!')
        elif self.options.merge:
            scaled_expt = []
            for expt in self.experiments:
                scaling = self.get_step_result(expt, StepType.SCALE)
                if scaling:
                    scaled_expt.append((expt, scaling))
            if scaled_expt:
                experiment, scaling = scaled_expt[0]
                name_label = ", ".join([expt.name for expt, res in scaled_expt])
                report = {
                    'title': f"{anom} Merging of {name_label}",
                    'kind': f'MX {anom} Analysis',
                    'score': round(scaling.get('quality.summary.score'), 2),
                    'details': reporting.merging_details(self)
                }
                report_file = str(self.options.directory / "report.html").replace(os.path.expanduser('~'), '~', 1)
                logger.info(f'- HTML report: {report_file}')
                reporting.save_report(report, self.options.directory)
            else:
                raise CommandFailed('No successful results to report on!')
        else:
            scaled_expt = []
            for expt in self.experiments:
                scaling = self.get_step_result(expt, StepType.SCALE)
                if scaling:
                    scaled_expt.append((expt, scaling))

            if scaled_expt:
                for expt, scaling in scaled_expt:
                    work_dir = self.options.working_directories[expt.identifier]
                    os.chdir(work_dir)
                    report_file = str(work_dir / "report.html").replace(os.path.expanduser('~'), '~', 1)
                    report = {
                        'title': f"{anom} Analysis of {expt.name}",
                        'kind': f'MX {anom} Analysis',
                        'score': round(scaling.get('quality.summary.score'), 2),
                        'details': reporting.single_details(self, expt)
                    }
                    logger.info(f'- HTML report: {report_file}')
                    reporting.save_report(report, self.options.directory)
            else:
                raise CommandFailed('No successful results to report on!')

