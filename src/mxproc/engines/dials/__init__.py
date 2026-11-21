from mxproc import Analysis
from mxproc.common import Step, Workflow
from mxproc.log import logger
from mxproc.engines.dials.parser import DIALSParser

STEP_INITIALIZE = Step("initialize", "Initialization")
STEP_SPOTS = Step("spots", "Spot Search", method="find_spots")
STEP_INDEX = Step("index", "Auto-Indexing & Refinement")
STEP_INTEGRATE = Step("integrate", "Integration of Intensities")


class DIALSAnalysis(Analysis):
    prefix = 'dials'
    workflows = {
        Workflow.PROCESS: (
            STEP_INITIALIZE,
            STEP_SPOTS,
            STEP_INDEX,
            STEP_INTEGRATE,
        ),
    }

    def initialize(self, **kwargs):
        results = {}
        logger.info('Preparing Working directories:')
        self.prepare_experiment_directories()
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            wildcard = str(experiment.directory / experiment.glob)

            logger.info_value(f' - {experiment.name}', str(directory))
            self.runner.run(f'dials.import {wildcard}', directory, desc=f"{experiment.name}: Importing data set")
        return results

    def find_spots(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]
            image_range = f'{experiment.frames[0][0]}-{experiment.frames[-1][1]}'
            self.runner.run(
                'dials.find_spots imported.expt',
                work_dir,
                desc=f'{experiment.name}: Finding strong spots in images {image_range}',
            )
            results[experiment.identifier] = DIALSParser.parse('spots.yml')

        return results

    def index(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]
            self.runner.run('dials.index imported.expt strong.refl', work_dir, desc="Auto-indexing")
            self.runner.run('dials.refine indexed.expt indexed.refl', work_dir, desc="Refining solution")
        return results

    def integrate(self, **kwargs) -> dict:
        results = {}
        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]
            self.runner.run('dials.integrate refined.expt refined.refl', work_dir, desc="Integrating images")

        return results
