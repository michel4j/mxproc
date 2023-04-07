import os
from mxproc import Analysis
from mxproc.command import run_command, CommandFailed
from mxproc.log import logger
from mxproc.engines.dials.parser import DIALSParser


class DIALSAnalysis(Analysis):
    prefix = 'dials'

    def initialize(self, **kwargs):
        results = {}
        logger.info('Preparing Working directories:')
        for experiment in self.experiments:
            directory = self.options.directory
            directory = directory if len(self.experiments) == 1 else directory / experiment.name
            directory.mkdir(exist_ok=True)
            self.options.working_directories[experiment.identifier] = directory

            wildcard = str(experiment.directory / experiment.glob)

            os.chdir(directory)
            logger.info_value(f' - {experiment.name}', str(directory))
            run_command(f'dials.import {wildcard}', desc=f"{experiment.name}: Importing data set")
        return results

    def find_spots(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])
            image_range = '{}-{}'.format(experiment.frames[0][0], experiment.frames[-1][1])
            run_command('dials.find_spots imported.expt', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
            results[experiment.identifier] = DIALSParser.parse('spots.yml')

        return results

    def index(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])
            run_command('dials.index imported.expt strong.refl', desc="Auto-indexing")
            run_command('dials.refine indexed.expt indexed.refl', desc="Refining solution")
        return results

    def integrate(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            os.chdir(self.options.working_directories[experiment.identifier])
            run_command('dials.integrate refined.expt refined.refl', desc="Integrating images")

        return results

    def report(self):
        pass

    def strategy(self, **kwargs):
        pass

    def symmetry(self, **kwargs):
        pass

    def scale(self, **kwargs):
        pass

    def export(self, **kwargs):
        pass

    def report(self, **kwargs):
        pass
