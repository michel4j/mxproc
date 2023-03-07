import os
from mxproc import Analysis
from mxproc.command import run_command, CommandFailed
from mxproc.engines.xds.parser import XDSParser, IndexProblem
from mxproc.log import logger
from mxproc.engines.xds import io
from mxproc.common import StateType, Status, AnalysisStep

MAX_INDEX_TRIES = 3


# Solutions to Indexing problems
INDEXING_SOLUTIONS = {
    IndexProblem.INDICES.value | IndexProblem.SUBTREES.value: {
        'min_spot_separation': 2,
        'cluster_radius': 1,
        'message': "Reducing minimum spot separation",
    },
    IndexProblem.POOR_SOLUTION.value: {
        'max_profile_error': (6.0, 4.0),
        'message': "Relaxing quality criteria to include more spots",
    },
    IndexProblem.REFINEMENT.value: {
        'refine_index': ('CELL', 'BEAM', 'ORIENTATION', 'AXIS'),
        'message': "Adjusting refinement parameters"
    },
}


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
                spot_range=experiment.frames,
            ))
            path_str = str(directory)
            path_str = path_str.replace(os.path.expanduser('~'), '~', 1)
            logger.info_value(f'{experiment.name}', path_str)
            results[experiment.identifier] = {
                'status': Status(state=StateType.SUCCESS, message="", flags=1),
                'details': directory
            }
        return results

    def find_spots(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            io.create_input_file(('XYCORR', 'INIT', 'COLSPOT'), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))
            image_range = '{}-{}'.format(experiment.frames[0][0], experiment.frames[-1][1])

            try:
                run_command('xds_par', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
                result = {
                    'status': Status(state=StateType.SUCCESS, message="", flags=1),
                    'details': XDSParser.parse('COLSPOT.LP'),
                }
            except CommandFailed as err:
                result = {
                    'status': Status(state=StateType.FAILURE, message=f"Command failed: {err}", flags=1),
                    'details': {}
                }
            results[experiment.identifier] = result
        return results

    def index(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            params = {
                'data_range': (experiment.frames[0][0], experiment.frames[-1][1]),
                'spot_range': experiment.frames,
            }

            # skip if spots not found
            if not self.step_succeeded(experiment, AnalysisStep.SPOTS):
                continue

            for i in range(MAX_INDEX_TRIES):
                parameters = io.XDSParameters(**params)
                io.create_input_file(('IDXREF',), experiment, parameters)

                try:
                    run_command('xds_par', desc=f'{experiment.name}: Auto-indexing and refinement of solution')
                    info = XDSParser.parse_index()
                    point_groups = ", ".join(info['details']['point_groups'])
                    reduced_cell = "{:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}".format(*info['details']['unit_cell'])
                    logger.info_value('Reduced Cell', reduced_cell)
                    logger.info_value('Compatible Point Groups', point_groups)
                except (CommandFailed, FileNotFoundError, KeyError) as err:
                    info = {
                        'status': Status(state=StateType.FAILURE, message=f"Command failed: {err}", flags=1),
                        'details': {}
                    }
                status = info['status']
                if status.state in [StateType.SUCCESS, StateType.WARNING] or status.flags & IndexProblem.COMMAND_ERROR.value:
                    break
                else:
                    logger.warning(status.message)
                    for test_flags, kw in INDEXING_SOLUTIONS.items():
                        if status.flags & test_flags:
                            message = kw.get('message', 'Trying different Parameters')
                            params.update(kw)
                            logger.warning(status.message)
                            logger.info(message)

            results[experiment.identifier] = info

            if status.state in [StateType.WARNING, StateType.FAILURE]:
                logger.warning_value(f"{experiment.name}:{status.state.name}", status.message)
            elif status.message:
                logger.info(f"{experiment.name}:{status.message}")
        return results

    def integrate(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            if not self.step_succeeded(experiment, AnalysisStep.INDEX):
                continue

            io.create_input_file(('DEFPIX', 'INTEGRATE', 'CORRECT',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))

            try:
                run_command('xds_par', desc=f'{experiment.name}: Integrating images')
                run_command('echo "XDS_ASCII.HKL" | xdsstat 20 3 > XDSSTAT.LP', desc=f'{experiment.name}: Gathering statistics')
                integration = XDSParser.parse('INTEGRATE.LP')
                correction = XDSParser.parse('CORRECT.LP')
                parameters = XDSParser.parse('GXPARM.XDS')
                stats = XDSParser.parse('XDSSTAT.LP')
            except CommandFailed as err:
                info = {
                    'status': Status(state=StateType.FAILURE, message=f"Command failed: {err}", flags=1),
                    'details': {}
                }
            else:
                details = {
                    'parameters': parameters,
                    'frame_summary': integration['scale_factors'],
                    'frame_statistics': stats['frame_statistics'],
                    'diff_statistics': stats['diff_statistics'],
                    'batch_statistics': integration['batches'],
                    'statistics_table': correction['statistics'],
                    'overall_summary': correction['statistics_overall'],
                    'lo_res_summary': correction['statistics'][0],
                    'hi_res_summary': correction['statistics'][-1],
                    'standard_errors': correction['standard_errors'],
                    'wilson_statistics': {
                        'plot': correction['wilson_plot'],
                        'line': correction['wilson_line'],
                    },
                    'correction_factors': correction['correction_factors']['factors'],
                }
                details['overall_summary'].update(correction['summary'])
                details['overall_summary'].update({
                    'sigma_asymptote': correction['correction_factors']['parameters']['sigma_asymptote'],
                })
                info = {
                    'status': Status(state=StateType.SUCCESS, message="", flags=1),
                    'details': details
                }
            results[experiment.identifier] = info

        return results
