import os
import subprocess

from mxproc import Analysis, run_command
from mxproc.engines.xds import io


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

            run_command('xds_par')
