import os
from mxproc import Analysis, run_command


class DIALSAnalysis(Analysis):

    def initialize(self, **kwargs):
        for experiment in self.experiments:
            directory = self.options.directory
            directory = directory if len(self.experiments) == 1 else directory / experiment.name
            directory.mkdir(exist_ok=True)
            self.options.working_directories[experiment.identifier] = directory

            wildcard = str(experiment.directory / experiment.glob)

            os.chdir(directory)
            run_command('dials.import', wildcard, desc=f"{experiment.name}: Importing data set")

    def find_spots(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)
            image_range = '{}-{}'.format(experiment.frames[0][0], experiment.frames[-1][1])
            run_command('dials.find_spots', 'imported.expt', desc=f'{experiment.name}: Finding strong spots in images {image_range}')
            # results[experiment.identifier] = DIALSParser.parse('COLSPOT.LP')

        return results

    def index(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)
            run_command('dials.index', 'imported.expt', 'strong.refl', desc="Auto-indexing")
            run_command('dials.refine', 'indexed.expt', 'indexed.refl', desc="Refining solution")
        return results

    def integrate(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            os.chdir(directory)
            run_command('dials.integrate', 'refined.expt', 'refined.refl', desc="Integrating images")

        return results

