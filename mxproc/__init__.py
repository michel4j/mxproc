import gzip
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Sequence, Union, Dict

import yaml

from mxproc import log
from mxproc.command import CommandFailed
from mxproc.common import StepType, InvalidAnalysisStep, StateType, Result, Workflow, logistic
from mxproc.xtal import load_multiple, Experiment
from mxproc.log import logger

__all__ = [
    "Analysis",
    "AnalysisOptions",
]

try:
    __version__ = version("mxproc")
except PackageNotFoundError:
    __version__ = "Dev"


# Links Analysis steps to the next step in the sequence
WORKFLOWS = {
    Workflow.SCREEN: {
        StepType.INITIALIZE: StepType.SPOTS,
        StepType.SPOTS: StepType.INDEX,
        StepType.INDEX: StepType.STRATEGY,
        StepType.STRATEGY: StepType.REPORT,
    },
    Workflow.PROCESS: {
        StepType.INITIALIZE: StepType.SPOTS,
        StepType.SPOTS: StepType.INDEX,
        StepType.INDEX: StepType.INTEGRATE,
        StepType.INTEGRATE: StepType.SYMMETRY,
        StepType.SYMMETRY: StepType.SCALE,
        StepType.SCALE: StepType.EXPORT,
        StepType.EXPORT: StepType.REPORT,
    }
}


@dataclass
class AnalysisOptions:
    files: Sequence[str]
    directory: Path
    working_directories: dict = field(default_factory=dict)
    screen: bool = False
    anomalous: bool = False
    merge: bool = True
    multi: bool = False
    extras: dict = field(default_factory=dict)


class Analysis(ABC):
    experiments: Sequence[Experiment]
    options: AnalysisOptions
    results: dict   # results for each experiment keyed by experiment identifier
    settings: dict  # Settings dictionary can be used for saving and recovering non-experiment specific information
    workflow: Workflow
    methods: dict   # Dictionary mapping steps to corresponding method

    prefix: str = 'proc'

    def __init__(
            self,
            *files: str,
            directory: Union[Path, str, None] = None,
            screen: bool = False,
            anomalous: bool = True,
            merge: bool = True
    ):
        """
        Data analysis objects
        :param files: image files corresponding to the datasets to process
        :param directory: top-level working directory, subdirectories may be created within for some processing options
        :param screen: Whether this is a screening analysis
        :param anomalous: Whether to process as anomalous data
        :param merge:  Whether to merge the datasets into a single output file or to keep them separate
        """

        # Prepare working directory
        if directory in ["", None]:
            index = 1
            directory = Path(f"{self.prefix}-{index}")
            while directory.exists():
                index += 1
                directory = Path(f"{self.prefix}-{index}")
        directory = Path(directory).absolute()

        self.experiments = load_multiple(files)
        merge &= len(self.experiments) > 1
        self.options = AnalysisOptions(
            files=files, directory=directory, screen=screen, anomalous=anomalous, merge=merge,
            multi=(len(self.experiments) > 1 and not merge)
        )

        self.results = {}
        self.settings = {}

        # workflow
        self.workflow = Workflow.SCREEN if self.options.screen else Workflow.PROCESS
        self.methods = {
            StepType.INITIALIZE: self.initialize,
            StepType.SPOTS: self.find_spots,
            StepType.INDEX: self.index,
            StepType.STRATEGY: self.strategy,
            StepType.INTEGRATE: self.integrate,
            StepType.SYMMETRY: self.symmetry,
            StepType.SCALE: self.scale,
            StepType.EXPORT: self.export,
            StepType.REPORT: self.report
        }

    def load(self, step: StepType):
        """
        Load an Analysis from a meta file and reset the state to it.
        :param step: analysis step corresponding to the saved metadata
        """
        meta_file = f'{step.slug()}.meta'

        try:
            with gzip.open(meta_file, 'rb') as handle:  # gzip compressed yaml file
                meta = yaml.load(handle, yaml.Loader)

            self.options = meta['options']
            self.experiments = meta['experiments']
            self.results = meta['results']
            self.settings = meta['settings']

        except FileNotFoundError:
            raise InvalidAnalysisStep('Checkpoint file missing. Must be loaded from working directory.')
        except (ValueError, TypeError, KeyError):
            raise InvalidAnalysisStep('Checkpoint file corrupted')

    def save(self, step: StepType, backup: bool = False):
        """
        Save analysis data to file
        :param backup: Whether to backup existing files, by default overwrite
        :param step: analysis step corresponding to the saved metadata
        """
        meta = {
            'options': self.options,
            'experiments': self.experiments,
            'results': self.results,
            'settings': self.settings
        }
        meta_file = self.options.directory / f'{step.slug()}.meta'

        # backup file if needed
        if meta_file.exists() and backup:
            meta_file.rename(f"{str(meta_file)}.bk")

        with gzip.open(meta_file, 'wb') as handle:  # gzip compressed yaml file
            yaml.dump(meta, handle, encoding='utf-8')

    def update_result(self, results: Dict[str, Result], step: StepType):
        """
        Update the results dictionary and save a checkpoint meta file

        :param results: dictionary of results keyed by the experiment identifier
        :param step: AnalysisStep
        """

        for identifier, result in results.items():
            experiment_results = self.results.get(identifier, {})
            if result.state in [StateType.SUCCESS, StateType.WARNING]:
                experiment_results.update({step.name: result})
                self.results[identifier] = experiment_results

        self.save(step)

    def get_step_result(self, expt: Experiment, step: StepType) -> Result | None:
        """
        Check if a given analysis step was successfully completed for a given experiment
        :param expt: the Experiment to check
        :param step: Analysis step to test
        """

        if step.name in self.results[expt.identifier]:
            return self.results[expt.identifier][step.name]

    @abstractmethod
    def score(self, expt: Experiment) -> float:
        """
        Calculate and return a data quality score for the specified experiment

        :param expt: experiment
        :return: float
        """

    def run(
            self,
            step: StepType = StepType.INITIALIZE,
            bootstrap: StepType | None = None,
            single: bool = False
    ):
        """
        Perform the analysis and gather the harvested results
        :param bootstrap: analysis step to use as a basis from the requested step. Must be higher than the previous workflow step
        :param single: Whether to run the full analysis from this step to the end of the workflow
        :param step: AnalysisStep to run
        """

        # If anything other than initialize, load the previous metadata and use that
        if step != StepType.INITIALIZE:
            bootstrap = step.prev() if bootstrap is None else bootstrap   # use previous if none
            self.load(bootstrap)

        start_time = time.time()
        header = f'MX Auto Proces (version: {__version__})'
        sub_header = f"{datetime.now().isoformat()} {self.workflow.desc()} [{len(self.experiments):d} dataset(s)]"
        logger.banner(header)
        logger.banner(sub_header, overline=False, line='-')

        while step is not None:
            # always go back to top-level working directory
            if self.options.directory.exists():
                os.chdir(self.options.directory)

            step_method = self.methods.get(step)
            try:
                logger.info_value(step.desc(), '', spacer='-')
                results = step_method()
            except CommandFailed as err:
                logger.error(f'Data Processing Failed at {step.name}: {err}. Aborting!')
                break
            else:
                # REPORTING step does not have results
                if step != StepType.REPORT and results:
                    self.update_result(results, step)

            # select the next step in the workflow
            step = None if single else WORKFLOWS[self.workflow].get(step)

        used_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        logger.banner(f'Processing Duration: {used_time}', line='-')

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the analysis of all experiments, should be ready for spot finding after this.
        :param kwargs: keyword argument to tweak initialization
        """
        ...

    @abstractmethod
    def find_spots(self, **kwargs) -> Dict[str, Result]:
        """
        Find spots, and prepare for indexing of all experiments
        :param kwargs: keyword argument to tweak spot search
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def index(self, **kwargs) -> Dict[str, Result]:
        """
        Perform indexing and refinement of all experiments
        :param kwargs: keyword argument to tweak indexing
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def strategy(self, **kwargs) -> Dict[str, Result]:
        """
        Perform Strategy determination and refinement of all experiments
        :param kwargs: keyword argument to tweak indexing
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def integrate(self, **kwargs) -> Dict[str, Result]:
        """
        Perform integration of all experiments.
        :param kwargs: keyword arguments for tweaking the integration settings.
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def symmetry(self, **kwargs) -> Dict[str, Result]:
        """
        Determination of Laue group symmetry and reindexing to the selected symmetry for all experiments.
        :param kwargs: keyword arguments for tweaking the symmetry
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def scale(self, **kwargs) -> Dict[str, Result]:
        """
        performs scaling on integrated datasets for all experiments
        :param kwargs: keyword arguments for tweaking the scaling
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def export(self, **kwargs) -> Dict[str, Result]:
        """
        Export the results of processing into various formats for all experiments
        :param kwargs: keyword arguments for tweaking the export
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def report(self, **kwargs) -> Dict[str, Result]:
        """
        Generate reports of the analysis in TXT and HTML formats for all experiments.
        :param kwargs: keyword arguments for tweaking the reporting
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...
