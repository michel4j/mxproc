from __future__ import annotations

import argparse
import gzip
import os
import time
import importlib
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Sequence, Dict, Tuple

import yaml

from mxproc import log
from mxproc.command import CommandFailed
from mxproc.common import StepType, InvalidAnalysisStep, StateType, Result, Workflow, logistic, parse_ranges
from mxproc.xtal import load_multiple, Experiment, Beam
from mxproc.log import logger

__all__ = [
    "Analysis",
    "AnalysisOptions",
    "Application"
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
    directory: Path
    files: Sequence[str] = ()
    working_directories: dict = field(default_factory=dict)
    screen: bool = False
    anomalous: bool = False
    optimize: bool = False
    merge: bool = True
    multi: bool = False
    extras: dict = field(default_factory=dict)
    beam_flux: float = 1e12
    beam_size: float = 100.
    beam_fwhm: Tuple[float, float] = (100., 100.)

    def get_beam(self):
        return Beam(self.beam_flux, *self.beam_fwhm, self.beam_size)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Analysis(ABC):
    experiments: Sequence[Experiment]
    options: AnalysisOptions
    results: dict   # results for each experiment keyed by experiment identifier
    settings: dict  # Settings dictionary can be used for saving and recovering non-experiment specific information
    workflow: Workflow
    methods: dict   # Dictionary mapping steps to corresponding method
    args: argparse.Namespace  # Contains parameters passed in through the command line.

    prefix: str = 'proc'

    def __init__(self, args: argparse.Namespace):
        """
        Data analysis objects
        :param args: arguments parsed from command line
        """

        self.args = args

        # Prepare working directory
        directory = self.args.dir
        if directory in ["", None]:
            index = 1
            directory = Path(f"{self.prefix}-{index}")
            while directory.exists():
                index += 1
                directory = Path(f"{self.prefix}-{index}")
        directory = Path(directory).absolute()

        self.experiments = load_multiple(self.args.images)
        self.options = AnalysisOptions(
            directory=directory, extras=self.get_extras(self.args), **self.get_options(self.args)
        )

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        log.log_to_file(str(directory / "auto.log"))

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

    def get_extras(self, args: argparse.Namespace) -> dict:
        """
        Extract Backend specific Parameters from arguments namespace object
        :param args: Namespace parsed from command line
        :return: dictionary
        """
        return {}

    def get_options(self, args: argparse.Namespace) -> dict:
        """
        Extract Parameters from arguments namespace object
        :param args: Namespace parsed from command line
        :return: dictionary suitable for initializing or updating a Options object
        """

        options = {
            'files': args.images,
            'screen': args.screen,
            'anomalous': args.anom,
            'optimize': args.optimize,
            'merge': len(self.experiments) > 1 and not args.multi,
            'multi': len(self.experiments) > 1 and args.multi,
            'beam_flux': args.beam_flux,
            'beam_fwhm': args.beam_fwhm,
            'beam_size': args.beam_size,
        }
        return {key: value for key, value in options.items() if value}

    def load(self, step: StepType):
        """
        Load an Analysis from a meta file and reset the state to it.
        :param step: analysis step corresponding to the saved metadata
        """
        meta_file = self.options.directory / f'{step.slug()}.meta'
        try:
            with gzip.open(meta_file, 'rb') as handle:  # gzip compressed yaml file
                meta = yaml.load(handle, yaml.Loader)

            self.options = meta['options']
            self.experiments = meta['experiments']
            self.results = meta['results']
            self.settings = meta['settings']

            # update options with current args
            self.options.update(**self.get_options(self.args))
            self.options.extras.update(**self.get_extras(self.args))

        except FileNotFoundError:
            raise InvalidAnalysisStep('Checkpoint file missing. Must be loaded from working directory.')
        except (ValueError, TypeError, KeyError):
            raise InvalidAnalysisStep('Checkpoint file corrupted')

    def save(self, step: StepType, backup: bool = False):
        """
        Save analysis data to file
        :param backup: Whether to back up existing files, by default overwrite
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
    ) -> int:
        """
        Perform the analysis and gather the harvested results
        :param bootstrap: analysis step to use as a basis from the requested step. Must be higher than the previous workflow step
        :param single: Whether to run the full analysis from this step to the end of the workflow
        :param step: AnalysisStep to run
        :return: valid python exit code, 0 = success, 1 = error, etc
        """

        exit_code = 0
        # If anything other than initialize, load the previous metadata and use that
        if step != StepType.INITIALIZE:
            # Find bootstrap step based on active workflow
            if bootstrap is None:
                for prev_step, next_step in WORKFLOWS[self.workflow].items():
                    if next_step == step:
                        bootstrap = prev_step
                        break
                else:
                    bootstrap = step.prev()
            self.load(bootstrap)

        start_time = time.time()
        header = f'MX Auto Proces (version: {__version__})'
        sub_header = f"{datetime.now().isoformat()} {self.workflow.desc()} [{len(self.experiments):d} dataset(s)]"
        logger.banner(header)
        logger.banner(sub_header, overline=False, line='-')

        while step is not None:
            # always go back to top-level working directory before running step
            if self.options.directory.exists():
                os.chdir(self.options.directory)

            step_method = self.methods.get(step)
            try:
                logger.info_value(step.desc(), '', spacer='-')
                results = step_method()
            except CommandFailed as err:
                logger.error(f'Data Processing Failed at {step.name}: {err}. Aborting!')
                exit_code = 1
                break
            else:
                # REPORTING step does not have results
                if step != StepType.REPORT and results:
                    self.update_result(results, step)

            # select the next step in the workflow
            step = None if single else WORKFLOWS[self.workflow].get(step)

        used_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        logger.banner(f'Processing Duration: {used_time}', line='-')
        return exit_code

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


class Application:
    def __init__(self, description: str, step: StepType = StepType.INITIALIZE):
        self.parser = argparse.ArgumentParser(description=f'MX AutoProcess: {description}')
        # main arguments
        self.parser.add_argument('images', nargs='*', help='Datasets to process. One frame per dataset.')
        self.parser.add_argument('-s', '--screen', help='Characterization and Strategy', action="store_true")
        self.parser.add_argument('-a', '--anom', help="Friedel's law False", action="store_true")
        self.parser.add_argument('-d', '--dir', type=str, help="Working Directory")
        self.parser.add_argument('-m', '--multi', help='Separate datasets scaled together', action="store_true")
        self.parser.add_argument('-u', '--use', type=str, default='XDS', help="Backend Engine. XDS | DIALS.")
        self.parser.add_argument('-1', '--single', action='store_true', default=False, help="Stop after a single step")
        self.parser.add_argument('-c', '--beam-center', type=float, nargs=2, help="Override Beam Center")

        # arguments for fine-tuning
        self.parser.add_argument('-f', '--frames', type=parse_ranges, help='Subset of Frames e.g: "45-400,410,420-450"')
        self.parser.add_argument('-r', '--resolution', type=float, help='Resolution limit')
        self.parser.add_argument('-o', '--optimize', help='Optimize re-processing', action="store_true")
        self.parser.add_argument('-g', '--spacegroup', type=int, help='Forced spacegroup number')

        # indexing arguments
        self.parser.add_argument('--min-sigma', type=float, help='Minimum I/Sigma of spots for indexing')
        self.parser.add_argument('--max-sigma', type=float, help="Maximum I/Sigma of spots for indexing")

        # beam arguments
        self.parser.add_argument('--beam-flux', type=float, help='Beam flux for dataset in ph/sec.')
        self.parser.add_argument('--beam-size', type=float, help="Beam aperture size")
        self.parser.add_argument('--beam-fwhm', type=float, nargs=2, help="Beam FWHM")
        self.step = step

    @staticmethod
    def get_engine(args: argparse.Namespace) -> Analysis:
        """
        Determine backend engine Analysis class based on engine name
        :param args:  Parsed arguments
        :return:  Analysis instance
        """
        name = args.use.upper()
        class_name = f'{name}Analysis'
        module_name = f'mxproc.engines.{name.lower()}'
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            proc = cls(args)
        except AttributeError as err:
            proc = None
            logger.error(f'Backend analysis engine {name} not found!')
            logger.exception(err)
        return proc

    def run(self):
        """
        Parse arguments and run analysis application
        """
        log.log_to_console(logging.DEBUG)
        args = self.parser.parse_args()
        proc = self.get_engine(args)
        if proc is not None:
            return proc.run(step=self.step, single=args.single)
        return 1

