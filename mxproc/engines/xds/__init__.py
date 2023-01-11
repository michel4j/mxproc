import os
from enum import Enum
from pathlib import Path
from mxproc import Analysis, run_command, TextParser
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
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)

            io.create_input_file(('XYCORR','INIT', 'COLSPOT'), experiment, io.XDSParameters(
                data_range=(experiment.frames[0][0], experiment.frames[-1][1]),
                spot_range=experiment.frames,
            ))

            output = run_command('xds_par')
            result = XDSParser.parse('COLSPOT.LP')
            print(result)
