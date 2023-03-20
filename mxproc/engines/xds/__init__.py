import os
import pprint
from pathlib import Path

from typing import Sequence, Tuple, Any, Dict
from mxproc import Analysis
from mxproc.command import run_command, CommandFailed
from mxproc.engines.xds.parser import XDSParser, IndexProblem, pointless
from mxproc.log import logger
from mxproc.engines.xds import io
from mxproc.common import StateType, Status, AnalysisStep, Flag, backup_files, generate_failure, Result, select_resolution
from mxproc.experiment import Experiment, Lattice, resolution_shells

MAX_INDEX_TRIES = 3
DEFAULT_MIN_SIGMA = 4
NUM_SCALE_SHELLS = 8


class IndexParamManager:
    """
    Stores and manages Indexing parameters across multiple trials, updating the parameters
    based on the problems detected during previous interation. Also keeps state so that degrees
    of suggested parameter changes would be based on the success or failure of previous attempts
    """

    degrees: dict                           # parameter degrees
    params: dict                            # previous parameters
    min_sigma: float                        # Maximum Spot Sigma Value
    max_sigma: float                        # Minimum Spot Sigma Value

    def __init__(
            self,
            data_range: Tuple[int, int],
            spot_range: Sequence[Tuple[int, int]],
            min_sigma: float = DEFAULT_MIN_SIGMA,
            max_sigma: float = 100
    ):
        self.degrees = {}
        self.params = {'spot_range': spot_range, "data_range": data_range}
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
            flags: int,
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

        if Flag.has(flags, IndexProblem.INVERTED_AXIS.value):
            self.params['invert_spindle'] = True
            messages.append('Inverting the rotation direction')
            if self.get_degree('invert_spindle') == 1:
                request_retry = True

        if Flag.has(flags, IndexProblem.WRONG_SPOT_PARAMETERS.value):
            self.params['spot_size'] = 0.5/self.get_degree('spot_size')
            messages.append("Reducing accepted spot size & separation")
            request_retry = True

        if Flag.has(flags, IndexProblem.LOW_INDEXED_FRACTION.value | IndexProblem.POOR_SOLUTION.value) and spot_range:
            self.params['spot_range'] = spot_range
            messages.append("Changing spot range")
            request_retry = True
        elif Flag.has(flags, IndexProblem.POOR_SOLUTION.value):
            self.params['error_scale'] = 2.0 * self.get_degree('error_scale')
            self.params['spot_separation'] = 1.0 - 0.25 ** self.get_degree('spot_separation')
            self.params['spot_size'] = 1.0 + 0.5 * self.get_degree('spot_size')
            messages.append("Relaxing quality criteria")
            request_retry = True

        if Flag.has(flags, IndexProblem.REFINEMENT_FAILED.value):
            refinement_flags = ('CELL', 'BEAM', 'ORIENTATION', 'AXIS',)[:-self.get_degree('refine_index')]
            if len(refinement_flags) > 1:
                self.params['refine_index'] = refinement_flags
                messages.append("Adjusting refinement parameters")
                request_retry = True

        if Flag.has(flags, IndexProblem.INSUFFICIENT_SPOTS.value):
            self.min_sigma = DEFAULT_MIN_SIGMA - self.get_degree('min_sigma')
            request_retry = True

        return request_retry, messages


def show_messages(label: str, messages: Sequence[str]):
    """
    Display a list of messages in the log
    :param label: Header label
    :param messages: sequence of messages

    """
    if messages:
        logger.warning(label)
        for message in messages:
            logger.warning(f' + {message}')


def find_lattice(lattice: Lattice, candidates: Sequence[dict]) -> Tuple[Lattice, tuple]:
    """
    Given a lattice, generate a lattice and reindex matrix, from the compatible spacegroup candidates
    :param lattice: search lattice
    :param candidates: candidates from XDS CORRECT
    :return: lattice, reindex_matrix
    """

    for candidate in candidates:
        if lattice.character == candidate['character']:
            a, b, c, alpha, beta, gamma = candidate['unit_cell']
            new_lattice = Lattice(spacegroup=lattice.spacegroup, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            return new_lattice, candidate['reindex_matrix']

    return None, None


class XDSAnalysis(Analysis):
    prefix = 'xds'

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

            io.create_input_file(('ALL',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames, skip_range=experiment.missing
            ))
            path_str = str(directory)
            path_str = path_str.replace(os.path.expanduser('~'), '~', 1)
            logger.info_value(f'{experiment.name}', path_str)
            results[experiment.identifier] = Result(status=Status(state=StateType.SUCCESS, flags=1), details=directory)
        return results

    def find_spots(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            io.create_input_file(('XYCORR', 'INIT', 'COLSPOT'), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames, skip_range=experiment.missing
            ))
            image_range = '{}-{}'.format(experiment.frames[0][0], experiment.frames[-1][1])

            try:
                run_command('xds_par', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
                result = Result(status=Status(state=StateType.SUCCESS), details=XDSParser.parse('COLSPOT.LP'))
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
            if not self.get_step_result(experiment, AnalysisStep.SPOTS):
                continue

            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            result = generate_failure('')
            logger.info(f'{experiment.name}:')
            param_manager = IndexParamManager(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames
            )

            for trial_number in range(MAX_INDEX_TRIES):
                backup_files('IDXREF.LP')
                result, retry_requested, retry_messages = self.autoindex_trial(experiment, param_manager, trial_number)
                if retry_requested:
                    show_messages(f'- Retrying Indexing:', retry_messages)
                else:
                    break

            # show final unit cell parameters
            if 'point_groups' in result.details:
                point_groups = ", ".join(result.details['point_groups'])
                reduced_cell = "{:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}".format(
                    *result.details['unit_cell']
                )
                logger.info_value('- Reduced Cell', reduced_cell)
                logger.info_value('- Compatible Point Groups', point_groups)

            results[experiment.identifier] = result
            status = result.status
            if status.state in [StateType.WARNING, StateType.FAILURE]:
                show_messages(f"- Final auto-indexing {status.state.name}:", status.messages)

        return results

    @staticmethod
    def autoindex_trial(experiment: Experiment, manager, trial: int) -> Tuple[Result, bool, Sequence[str]]:
        """
        Run one trials of auto indexing
        :param experiment: target Experiment for the trial
        :param manager: Parameter manager
        :param trial: integer representing the trial number
        :return:
        """

        parameters = io.XDSParameters(**manager.get_parameters())
        io.filter_spots(min_sigma=manager.min_sigma, max_sigma=manager.max_sigma)
        io.create_input_file(('IDXREF',), experiment, parameters)
        try:
            run_command('xds_par', desc=f'- Attempt #{trial + 1} of auto-indexing')
            status, details = XDSParser.parse_index()
            result = Result(status=status, details=details)
        except (CommandFailed, FileNotFoundError, KeyError) as err:
            result = generate_failure(f"Command failed: {err}")

        status = result.status
        retry_messages = []
        request_retry = not (
            status.state in [StateType.SUCCESS, StateType.WARNING] or
            Flag.has(status.flags, IndexProblem.SOFTWARE_ERROR.value)
        )

        if request_retry:
            show_messages(f'- Indexing Problems:', status.messages)
            request_retry, retry_messages = manager.update_parameters(
                status.flags, spot_range=result.details['spots']['best_range']
            )

        return result, request_retry, retry_messages

    def integrate(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            if not self.get_step_result(experiment, AnalysisStep.INDEX):
                continue

            logger.info(f'{experiment.name}:')
            io.create_input_file(('DEFPIX', 'INTEGRATE', 'CORRECT',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames, skip_range=experiment.missing
            ))
            frame_range = f'{experiment.frames[0][0]}-{experiment.frames[-1][1]}'
            try:
                run_command('xds_par', desc=f'- Integrating frames {frame_range}')
                integration = XDSParser.parse('INTEGRATE.LP')
                correction = XDSParser.parse('CORRECT.LP')
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                details = {
                    'batches': integration['batches'],
                    'frames': integration['frames'],
                    'lattices': correction['lattices'],
                    'quality': {
                        'statistics': correction['statistics'],
                        'overall': correction['statistics_overall'],
                        'errors': correction['standard_errors'],
                    },
                }
                result = Result(status=Status(state=StateType.SUCCESS, flags=1), details=details)
            results[experiment.identifier] = result

        return results

    def symmetry(self, **kwargs):
        results = {}
        reference = self.experiments[0]

        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            if self.get_step_result(experiment, AnalysisStep.INTEGRATE):
                logger.info(f'{experiment.name}:')
                try:
                    run_command('pointless xdsin INTEGRATE.HKL xmlout pointless.xml',  desc=f'- Determining symmetry')
                    details = pointless.parse_pointless('pointless.xml')
                    experiment.lattice = details['lattice']
                except CommandFailed as err:
                    result = generate_failure(f"Command failed: {err}")
                else:
                    result = Result(status=Status(state=StateType.SUCCESS, flags=1), details=details)
                    lattice_message = f'{experiment.lattice.name} - #{experiment.lattice.spacegroup}'
                    logger.info_value(f'- Selected {details["type"]}', lattice_message)
                    if experiment.lattice.spacegroup > reference.lattice.spacegroup:
                        reference = experiment
                results[experiment.identifier] = result

        for experiment in self.experiments:
            if experiment.identifier not in results:
                continue
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            result = self.apply_symmetry(experiment, reference)
            result.details['symmetry'] = results[experiment.identifier]
            results[experiment.identifier] = result

            unit_cell = "{:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}".format(
                *result.details['parameters']['unit_cell']
            )
            i_sigma_a = result.details['quality']['overall']['i_sigma_a']
            lo_r_meas = result.details['quality']['statistics'][0]['r_meas']
            lo_i_sigma = result.details['quality']['statistics'][0]['i_sigma']
            completeness = result.details['quality']['overall']['completeness']
            resolution = result.details['quality']['overall']['resolution']
            resolution_method = result.details['quality']['overall']['resolution_method']

            logger.info_value(f'- Refined Cell:', unit_cell)
            logger.info_value(f'- Low-res R-meas:', f'{lo_r_meas:0.1f} %')
            logger.info_value(f'- Low-res I/Sigma:', f'{lo_i_sigma:0.1f}')
            logger.info_value(f'- Completeness:', f'{completeness:0.1f} %')
            logger.info_value(f'- I/Sigma(I) Asymptote [ISa]:', f'{i_sigma_a:0.1f}')
            logger.info_value(f'- Resolution limit (by {resolution_method}):', f'{resolution:0.2f}')
        return results

    def apply_symmetry(self, experiment: Experiment, reference: Experiment) -> Result:
        """
        Apply specified symmetry to the experiment and reindex the data
        :param experiment: Experiment instance
        :param reference: Reference experiment
        """
        logger.info(f'{experiment.name}:')
        integrate_result = self.get_step_result(experiment, AnalysisStep.INTEGRATE)
        reindex_lattice, reindex_matrix = find_lattice(reference.lattice, integrate_result.details['lattices'])
        if reindex_lattice:
            if reference != experiment:
                directory = self.options.working_directories[experiment.identifier]
                reference_directory = self.options.working_directories[reference.identifier]
                reference_data = Path(os.path.relpath((reference_directory / "XDS_ASCII.HKL"), directory))
            else:
                reference_data = None

            try:
                io.create_input_file(('CORRECT',), experiment, io.XDSParameters(
                    data_range=(experiment.frames[0][0], experiment.frames[-1][1]), reference=reference_data,
                    spot_range=experiment.frames, reindex=reindex_matrix, lattice=reindex_lattice,
                    skip_range=experiment.missing
                ))
                run_command(
                    'xds_par', desc=f'- Applying symmetry {reindex_lattice.name} - #{reference.lattice.spacegroup}'
                )
                run_command('echo "XDS_ASCII.HKL" | xdsstat 20 3 > XDSSTAT.LP', desc=f'- Gathering extra statistics')
                correction = XDSParser.parse('CORRECT.LP')
                parameters = XDSParser.parse('GXPARM.XDS')
                stats = XDSParser.parse('XDSSTAT.LP')
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                resolution, resolution_method = select_resolution(correction['statistics'])
                details = {
                    'parameters': parameters['parameters'],
                    'lattices': correction['lattices'],
                    'quality': {
                        'frames': stats['frame_statistics'],
                        'differences': stats['diff_statistics'],
                        'statistics': correction['statistics'],
                        'overall': correction['statistics_overall'],
                        'errors': correction['standard_errors'],
                    },
                    'wilson': {
                        'plot': correction['wilson_plot'],
                        'line': correction['wilson_line'],
                    },
                    'correction_factors': [
                        item['chi_sqr']
                        for item in correction['correction_factors']['factors']
                    ],
                }
                details['quality']['overall'].update(
                    **correction['summary'],
                    i_sigma_a=correction['correction_factors']['parameters']['i_sigma_a'],
                    resolution=resolution, resolution_method=resolution_method,
                )
                result = Result(status=Status(state=StateType.SUCCESS), details=details)
        else:
            logger.warning(f'{experiment.name} not compatible with selected spacegroup #{reference.lattice.spacegroup}')
            result = generate_failure(f"Incompatible Lattice type: {reference.lattice.character}")
        return result

    def scale(self, **kwargs):
        scalable_experiments = [
            experiment for experiment in self.experiments if self.get_step_result(experiment, AnalysisStep.SYMMETRY)
        ]
        scale_configs = []
        if self.options.merge:
            # prepare merge parameters
            inputs = []
            resolutions = []
            for experiment in scalable_experiments:
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, AnalysisStep.SYMMETRY)
                input_file = Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory))
                resolution = symmetry_result.details['quality']['overall']['resolution']
                inputs.append({'input_file': input_file, 'resolution': resolution})
                resolutions.append(resolution)

            # one configuration for merging
            scale_configs.append({
                'anomalous': self.options.anomalous,
                'shells': resolution_shells(min(resolutions), NUM_SCALE_SHELLS),
                'output_file': "XSCALE.HKL",
                'inputs': inputs,
            })
            backup_files('XSCALE.HKL', 'XSCALE.LP')
        else:
            for experiment in scalable_experiments:
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, AnalysisStep.SYMMETRY)
                input_file = Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory))
                resolution = symmetry_result.details['quality']['overall']['resolution']
                scale_configs.append({
                    'anomalous': self.options.anomalous,
                    'shells': resolution_shells(resolution, NUM_SCALE_SHELLS),
                    'output_file':  directory / "XSCALE.HKL",
                    'inputs': [{'input_file': input_file, 'resolution': resolution}],
                })

        try:
            os.chdir(self.options.directory)
            io.write_xscale_input(scale_configs)

            run_command('xscale_par', f'Scaling {len(scalable_experiments)} dataset(s)')
            details = XDSParser.parse('XSCALE.LP')
        except CommandFailed as err:
            result = generate_failure(f'Command Failed: {err}')
        else:
            #pprint.pprint(details)
            pass

