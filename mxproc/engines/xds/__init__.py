import os
from enum import Enum
from pathlib import Path
from mxproc import Analysis, run_command, TextParser, logger
from mxproc.engines.xds import io


class IndexFailure(Enum):
    NONE = 0
    LOW_DIMENSION = 1
    LOW_FRACTION = 2
    FEW_SPOTS = 3
    POOR_SOLUTION = 4
    REFINEMENT = 4
    TERMINATED = 5


FAILURES = {
    r'CANNOT CONTINUE WITH A TWO-DIMENSIONAL': IndexFailure.LOW_DIMENSION,
    r'DIMENSION OF DIFFERENCE VECTOR SET LESS THAN \d+.': IndexFailure.LOW_DIMENSION,
    r'INSUFFICIENT NUMBER OF ACCEPTED SPOTS.': IndexFailure.FEW_SPOTS,
    r'SOLUTION IS INACCURATE': IndexFailure.POOR_SOLUTION,
    r'RETURN CODE IS IER= \s+\d+': IndexFailure.REFINEMENT,
    r'CANNOT INDEX REFLECTIONS': IndexFailure.TERMINATED,
    r'^INSUFFICIENT PERCENTAGE .+ OF INDEXED REFLECTIONS': IndexFailure.LOW_FRACTION,
}

DATA_PATH = Path(__file__).parent / "data"


class XDSParser(TextParser):
    LEXICON = {
        "COLSPOT.LP": DATA_PATH / "spots.yml",
        "IDXREF.LP": DATA_PATH / "idxref.yml",
        "INTEGRATE.LP": DATA_PATH / "integrate.yml",
        "CORRECT.LP": DATA_PATH / "correct.yml",
        "XSCALE.LP": DATA_PATH / "xscale.yml",
        "XDSSTAT.LP": DATA_PATH / "xdsstat.yml",
        "XPARM.XDS": DATA_PATH / "xparm.yml",
        "GXPARM.XDS": DATA_PATH / "xparm.yml"
    }


class XDSAnalysis(Analysis):

    def initialize(self, **kwargs):
        for experiment in self.experiments:
            # create sub-directories if multiple processing
            directory = self.options.directory
            directory = directory if len(self.experiments) == 1 else directory / experiment.name
            directory.mkdir(exist_ok=True)
            self.options.working_directories[experiment.identifier] = directory

            os.chdir(self.options.working_directories[experiment.identifier])

            io.create_input_file(('ALL',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))

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
            run_command('xds_par', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
            results[experiment.identifier] = XDSParser.parse('COLSPOT.LP')

        return results

    def index(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            io.create_input_file(('IDXREF',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))

            run_command('xds_par', desc=f'{experiment.name}: Auto-indexing and refinement of solution')
            results[experiment.identifier] = XDSParser.parse('IDXREF.LP')

        return results

    def integrate(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            io.create_input_file(('DEFPIX', 'INTEGRATE', 'CORRECT',), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))

            run_command('xds_par', desc=f'{experiment.name}: Integrating images')
            run_command('echo "XDS_ASCII.HKL" | xdsstat 20 3 > XDSSTAT.LP', desc=f'{experiment.name}: Gathering statistics')
            integration = XDSParser.parse('INTEGRATE.LP')
            correction = XDSParser.parse('CORRECT.LP')
            parameters = XDSParser.parse('GXPARM.XDS')
            stats = XDSParser.parse('XDSSTAT.LP')

            results[experiment.identifier] = {
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
            results[experiment.identifier]['overall_summary'].update(correction['summary'])
            results[experiment.identifier]['overall_summary'].update({
                'sigma_asymptote': correction['correction_factors']['parameters']['sigma_asymptote'],
            })

        return results
