import gzip
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
from mxproc.common import AnalysisStep, InvalidAnalysisStep, StateType, Result
from mxproc.experiment import load_multiple, Experiment
from mxproc.log import logger

__all__ = [
    "Analysis",
    "AnalysisOptions",
]

try:
    __version__ = version("mxproc")
except PackageNotFoundError:
    __version__ = "Dev"


@dataclass
class AnalysisOptions:
    files: Sequence[str]
    directory: Path
    working_directories: dict = field(default_factory=dict)
    anomalous: bool = False
    merge: bool = True


class Analysis(ABC):
    experiments: Sequence[Experiment]
    options: AnalysisOptions
    results: dict  # results for each experiment keyed by experiment identifier
    prefix: str = 'proc'

    def __init__(self, *files, directory: Union[Path, str, None] = None, anomalous: bool = False, merge: bool = True):
        """
        Data analysis objects
        :param files: image files corresponding to the datasets to process
        :param directory: top-level working directory, subdirectories may be created within for some processing options
        :param anomalous: Whether to process as anomalous data
        :param merge:  Whether to merge the datasets into a single output file or to keep them separate
        """

        # Prepare directory
        if directory in ["", None]:
            index = 1
            directory = Path(f"{self.prefix}-{index}")
            while directory.exists():
                index += 1
                directory = Path(f"{self.prefix}-{index}")

        self.options = AnalysisOptions(
            files=files, directory=Path(directory).absolute(), anomalous=anomalous, merge=merge
        )
        self.experiments = load_multiple(files)
        self.results = {}

    def load(self, step: AnalysisStep):
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

        except FileNotFoundError:
            raise InvalidAnalysisStep('Checkpoint file missing. Must be loaded from working directory.')
        except (ValueError, TypeError, KeyError):
            raise InvalidAnalysisStep('Checkpoint file corrupted')

    def save(self, step: AnalysisStep, backup: bool = False):
        """
        Save analysis data to file
        :param backup: Whether to backuup existing files, by default overwrite
        :param step: analysis step corresponding to the saved metadata
        """
        meta = {
            'options': self.options,
            'experiments': self.experiments,
            'results': self.results
        }
        meta_file = self.options.directory / f'{step.slug()}.meta'

        # backup file if needed
        if meta_file.exists() and backup:
            meta_file.rename(f"{str(meta_file)}.bk")

        with gzip.open(meta_file, 'wb') as handle:  # gzip compressed yaml file
            yaml.dump(meta, handle, encoding='utf-8')

    def update_result(self, results: Dict[str, Result], step: AnalysisStep):
        """
        Update the results dictionary and save a checkpoint meta file

        :param results: dictionary of results keyed by the experiment identifier
        :param step: AnalysisStep
        """

        for identifier, result in results.items():
            experiment_results = self.results.get(identifier, {})
            status = result.status
            if status.state in [StateType.SUCCESS, StateType.WARNING]:
                experiment_results.update({step.name: result})
                self.results[identifier] = experiment_results

        self.save(step)

    def get_step_result(self, expt: Experiment, step: AnalysisStep) -> Result | None:
        """
        Check if a given analysis step was successfully completed for a given experiment
        :param expt: the Experiment to check
        :param step: Analysis step to test
        """

        if step.name in self.results[expt.identifier]:
            return self.results[expt.identifier][step.name]

    def run(self, next_step: AnalysisStep = AnalysisStep.INITIALIZE, bootstrap: Union[AnalysisStep, None] = None,
            complete: bool = True):
        """
        Perform the analysis and gather the harvested results
        :param bootstrap: analysis step to use as a basis from the requested step. Must be higher than the previous workflow step
        :param complete: Whether to run the full analysis from this step to the end of the workflow
        :param next_step: AnalysisStep to run
        """

        workflow = {
            AnalysisStep.INITIALIZE: self.initialize,
            AnalysisStep.SPOTS: self.find_spots,
            AnalysisStep.INDEX: self.index,
            AnalysisStep.INTEGRATE: self.integrate,
            AnalysisStep.SYMMETRY: self.symmetry,
            AnalysisStep.SCALE: self.scale,
            AnalysisStep.QUALITY: self.quality,
            AnalysisStep.EXPORT: self.export,
            AnalysisStep.REPORT: self.report
        }

        # If anything other than initialize, load the previous metadata and use that
        if next_step.value > 0:
            if bootstrap is None:
                # load meta data from previous step
                self.load(AnalysisStep(next_step.value - 1))
            elif bootstrap.value >= next_step.value - 1:
                # load if it is at least greater than the previous step from the requested one
                self.load(bootstrap)

        start_time = time.time()
        header = f'MX Auto Processing (version: {__version__})'
        sub_header = "{} [{:d} dataset(s)]".format(datetime.now().isoformat(), len(self.experiments))
        logger.banner(header)
        logger.banner(sub_header, overline=False, line='-')

        for step, step_method in workflow.items():
            # skip all steps prior to requested_step
            if step < next_step:
                continue

            # do not run subsequent steps if complete is False
            if step > next_step and not complete:
                break

            try:
                logger.info_value(step.desc(), '', spacer='-')
                results = step_method()
            except CommandFailed as err:
                logger.error(f'Data Processing Failed at {step.name}: {err}. Aborting!')
                break
            else:
                # REPORTING step does not have results
                if step != AnalysisStep.REPORT and results:
                    self.update_result(results, step)

        used_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        logger.banner(f'Processing done. Duration: {used_time}', line='-')

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

    # @abstractmethod
    def scale(self, **kwargs) -> Dict[str, Result]:
        """
        performs scaling on integrated datasets for all experiments
        :param kwargs: keyword arguments for tweaking the scaling
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    # @abstractmethod
    def export(self, **kwargs) -> Dict[str, Result]:
        """
        Export the results of processing into various formats for all experiments
        :param kwargs: keyword arguments for tweaking the export
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    # @abstractmethod
    def report(self, **kwargs) -> Dict[str, Result]:
        """
        Generate reports of the analysis in TXT and HTML formats for all experiments.
        :param kwargs: keyword arguments for tweaking the reporting
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    # @abstractmethod
    def quality(self, **kwargs) -> Dict[str, Result]:
        """
        Check data quality of all experiments.
        :param kwargs: keyword arguments for tweaking quality check
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...
