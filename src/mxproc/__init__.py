from __future__ import annotations

import argparse
import gzip
import os
import time
import importlib
import logging

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import ClassVar, Mapping, Sequence, Dict, Tuple

import yaml

from mxproc import log
from mxproc.command import CommandFailed, CommandRunner, SlurmRunner
from mxproc.common import (
    Step, InvalidAnalysisStep, StateType, Result, Workflow, parse_ranges, fix_permissions,
    parse_cluster
)
from mxproc.xtal import load_multiple, Experiment, Beam
from mxproc.log import logger

__all__ = [
    "Analysis",
    "AnalysisOptions",
    "Application",
    "Step",
    "peek_checkpoint_engine",
]

try:
    __version__ = version("mxproc")
except PackageNotFoundError:
    __version__ = "Dev"


def _checkpoint_path(directory: Path, step: str | Step | None = None) -> Path:
    realm = 'latest' if step is None else (step.slug if isinstance(step, Step) else step)
    return directory / f'{realm}.meta'


def peek_checkpoint_engine(directory: Path, step: str | Step | None = None) -> str | None:
    """
    Read the owning engine identifier from a checkpoint file if present and readable.
    Returns None when the checkpoint is missing or unreadable.
    """
    meta_file = _checkpoint_path(directory, step)
    if not meta_file.exists():
        return None
    try:
        with gzip.open(meta_file, 'rb') as handle:
            meta = yaml.load(handle, yaml.Loader)
    except Exception:
        return None
    if isinstance(meta, dict):
        engine = meta.get('engine')
        if isinstance(engine, str) and engine:
            return engine
    return None


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
    results: dict  # results for each experiment keyed by experiment identifier, then step slug
    settings: dict  # Settings dictionary can be used for saving and recovering non-experiment specific information
    workflow: Workflow
    args: argparse.Namespace  # Contains parameters passed in through the command line.

    prefix: str = 'proc'
    workflows: ClassVar[Mapping[Workflow, Sequence[Step]]] = {}

    def __init__(self, args: argparse.Namespace, runner: CommandRunner | None = None):
        """
        Data analysis objects
        :param args: arguments parsed from command line
        :param runner: optional CommandRunner seam for command execution
        """

        self.args = args
        self.experiments = ()
        self.runner: CommandRunner = runner if runner is not None else SlurmRunner()

        # prepare and load experiment
        if args.images:
            self.experiments = load_multiple(self.args.images)
            directory = self.prepare_directory(self.args.dir)
        else:
            # Load from current working directory
            directory = Path(self.args.dir).absolute()
            logger.info_value('Resuming', str(directory))

        self.options = AnalysisOptions(
            directory=directory, extras=self.get_extras(self.args), **self.get_options(self.args)
        )

        log.log_to_file(str(directory / "auto.log"))

        self.results = {}
        self.settings = {}

        # workflow
        self.workflow = Workflow.SCREEN if self.options.screen else Workflow.PROCESS

    @classmethod
    def get_steps(cls, workflow: Workflow) -> tuple[Step, ...]:
        """Return ordered steps for a workflow, or raise InvalidAnalysisStep if unsupported."""
        steps = cls.workflows.get(workflow)
        if not steps:
            supported = ', '.join(wf.name for wf in cls.workflows) or 'none'
            raise InvalidAnalysisStep(
                f"Workflow '{workflow.name}' is not supported by engine '{cls.prefix}'. "
                f"Supported workflows: {supported}."
            )
        return tuple(steps)

    @classmethod
    def resolve_step(cls, workflow: Workflow, step: str | Step | None = None) -> Step:
        """Resolve a step slug or Step instance within the given workflow."""
        steps = cls.get_steps(workflow)
        if step is None:
            return steps[0]
        target_slug = step.slug if isinstance(step, Step) else step.strip().lower()
        for candidate in steps:
            if candidate.slug == target_slug:
                return candidate
        available = ', '.join(s.slug for s in steps)
        raise InvalidAnalysisStep(
            f"Step '{target_slug}' is not supported by engine '{cls.prefix}' "
            f"for workflow '{workflow.name}'. Available steps: {available}."
        )

    @classmethod
    def _has_step_in_any_workflow(cls, step: str | Step) -> bool:
        target_slug = step.slug if isinstance(step, Step) else step.strip().lower()
        return any(
            candidate.slug == target_slug
            for steps in cls.workflows.values()
            for candidate in steps
        )

    @classmethod
    def format_steps(cls) -> str:
        """Format available workflows and their steps for CLI display."""
        lines = [f"Engine: {cls.prefix}"]
        if not cls.workflows:
            lines.append("  (no workflows defined)")
            return "\n".join(lines)
        for workflow, steps in cls.workflows.items():
            lines.append(f"  {workflow.name} ({workflow.desc()}):")
            for idx, step in enumerate(steps, start=1):
                lines.append(f"    {idx}. {step.slug:<12} {step.description}")
        return "\n".join(lines)

    def prepare_directory(self, directory: str) -> Path:
        """
        Prepare the working directory for the analysis, if the directory is provided and already exists,
        do not create a new one, otherwise create a new directory with a unique name.

        :param directory: path to the working directory
        :return: Path object for working directory. Directory exists at this point
        """

        top_level = Path(directory).absolute()
        path = top_level

        if not directory:
            # No directory string provided, create a new directory
            index = 0
            path = top_level / f"{self.prefix}-{index}"
            while path.exists():
                index += 1
                path = top_level / f"{self.prefix}-{index}"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def prepare_experiment_directories(self) -> Dict[str, Path]:
        """
        Create per-experiment working directories and register them in options.working_directories.
        """
        for experiment in self.experiments:
            directory = self.options.directory
            directory = directory if len(self.experiments) == 1 else directory / experiment.name
            directory.mkdir(parents=True, exist_ok=True)
            self.options.working_directories[experiment.identifier] = directory
        return self.options.working_directories

    def get_extras(self, args: argparse.Namespace) -> dict:
        """
        Extract engine-specific parameters from arguments namespace object
        :param args: Namespace parsed from command line
        :return: dictionary
        """
        return {}

    def get_options(self, args: argparse.Namespace) -> dict:
        """
        Extract parameters from arguments namespace object
        :param args: Namespace parsed from command line
        :return: dictionary suitable for initializing or updating an AnalysisOptions object
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

    def load(self, step: str | Step | None = None):
        """
        Load an Analysis from a checkpoint file and reset state to it.
        :param step: step slug or Step corresponding to the saved checkpoint; reads latest if None
        """

        meta_file = _checkpoint_path(self.options.directory, step)

        try:
            with gzip.open(meta_file, 'rb') as handle:  # gzip compressed yaml file
                meta = yaml.load(handle, yaml.Loader)

            if not isinstance(meta, dict):
                raise ValueError('Checkpoint payload must be a mapping')

            meta_engine = meta.get('engine')
            if not meta_engine:
                raise InvalidAnalysisStep(
                    'Checkpoint file missing engine metadata (incompatible legacy checkpoint).'
                )
            if str(meta_engine).lower() != self.prefix.lower():
                raise InvalidAnalysisStep(
                    f"Checkpoint belongs to engine '{meta_engine}', cannot resume with '{self.prefix}'."
                )

            self.options = meta['options']
            self.experiments = meta['experiments']
            self.results = meta['results']
            self.settings = meta['settings']

            meta_workflow = meta.get('workflow')
            if not self.args.screen and meta_workflow:
                self.workflow = (
                    meta_workflow
                    if isinstance(meta_workflow, Workflow)
                    else Workflow[str(meta_workflow)]
                )

            # update options with current args
            self.options.update(**self.get_options(self.args))
            self.options.extras.update(**self.get_extras(self.args))

        except FileNotFoundError:
            raise InvalidAnalysisStep('Checkpoint file missing. Must be loaded from working directory.')
        except InvalidAnalysisStep:
            raise
        except (ValueError, TypeError, KeyError, OSError):
            raise InvalidAnalysisStep('Checkpoint file corrupted')

    def save(self, step: Step, backup: bool = False):
        """
        Save analysis state to a checkpoint file.
        :param step: Step corresponding to the saved checkpoint
        :param backup: Whether to back up existing checkpoint file; by default overwrite
        """
        meta = {
            'engine': self.prefix,
            'workflow': self.workflow.name,
            'step': step.slug,
            'options': self.options,
            'experiments': self.experiments,
            'results': self.results,
            'settings': self.settings
        }

        meta_file = _checkpoint_path(self.options.directory, step)
        latest_file = _checkpoint_path(self.options.directory)

        # backup file if needed
        if meta_file.exists() and backup:
            meta_file.rename(f"{str(meta_file)}.bk")

        with gzip.open(meta_file, 'wb') as handle:  # gzip compressed yaml file
            yaml.dump(meta, handle, encoding='utf-8')

        latest_file.unlink(missing_ok=True)
        latest_file.symlink_to(meta_file.name)

    def update_result(self, results: Dict[str, Result] | None, step: Step):
        """
        Update the results dictionary and save a checkpoint file.

        :param results: dictionary of results keyed by the experiment identifier
        :param step: completed Step
        """

        if results:
            for identifier, result in results.items():
                experiment_results = self.results.get(identifier, {})
                if result.state in (StateType.SUCCESS, StateType.WARNING):
                    experiment_results[step.slug] = result
                    self.results[identifier] = experiment_results

        self.save(step)

    def get_step_result(self, expt: Experiment, step: str | Step) -> Result | None:
        """
        Return the saved Result for an experiment and step slug, or None if absent.
        :param expt: the Experiment to check
        :param step: Step or step slug to look up
        """
        slug = step.slug if isinstance(step, Step) else step
        return self.results.get(expt.identifier, {}).get(slug)

    def run(
            self,
            step: str | Step | None = None,
            bootstrap: str | Step | None = None,
            single: bool = False
    ) -> int:
        """
        Perform the analysis and gather the harvested results.
        :param step: starting Step or step slug (defaults to workflow entry step)
        :param bootstrap: checkpoint step slug/Step to load before resuming; defaults to latest
        :param single: Whether to run only a single step or continue to the end of the workflow
        :return: valid python exit code, 0 = success, 1 = error, etc
        """

        exit_code = 0
        initial_workflow = self.workflow

        # Validate workflow/step before loading checkpoints so unsupported steps fail loudly.
        if step is None:
            steps = self.get_steps(self.workflow)
            resolved = steps[0]
            start_idx = 0
        else:
            steps = self.get_steps(self.workflow)
            try:
                resolved = self.resolve_step(self.workflow, step)
                start_idx = steps.index(resolved)
                step_in_initial = True
            except InvalidAnalysisStep:
                if self.args.screen or not self._has_step_in_any_workflow(step):
                    raise
                step_in_initial = False
                start_idx = -1

            if not step_in_initial or start_idx > 0:
                self.load(bootstrap)
                try:
                    steps = self.get_steps(self.workflow)
                    resolved = self.resolve_step(self.workflow, step)
                except InvalidAnalysisStep:
                    if step_in_initial:
                        self.workflow = initial_workflow
                        steps = self.get_steps(self.workflow)
                        resolved = self.resolve_step(self.workflow, step)
                    else:
                        raise
                start_idx = steps.index(resolved)

        start_time = time.time()
        header = f'MX Auto Proces (version: {__version__})'
        sub_header = f"{datetime.now().isoformat()} {self.workflow.desc()} [{len(self.experiments):d} dataset(s)]"
        logger.banner(header)
        logger.banner(sub_header, overline=False, line='-')
        try:
            for current_step in steps[start_idx:]:
                step_method = getattr(self, current_step.method_name)
                try:
                    logger.info_value(current_step.description, '', spacer='-')
                    results = step_method()
                except CommandFailed as err:
                    logger.error(f'Failed at {current_step.slug}: {err}. Aborting!')
                    exit_code = 1
                    break
                else:
                    if current_step.checkpoint:
                        self.update_result(results, current_step)

                if single:
                    break
        finally:
            if self.options.directory.exists():
                with os.scandir(self.options.directory):  # reload directory cache
                    pass
                fix_permissions(self.options.directory)

        used_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        logger.banner(f'Processing Duration: {used_time}', line='-')
        return exit_code


class Application:
    def __init__(self, description: str, step: str | None = None, screen: bool = False):
        self.parser = argparse.ArgumentParser(description=f'MX AutoProcess: {description}')
        # main arguments
        self.parser.add_argument('images', nargs='*', help='Datasets to process. One frame per dataset.')
        self.parser.add_argument('-s', '--screen', help='Characterization and Strategy', action="store_true", default=screen)
        self.parser.add_argument('-a', '--anom', help="Friedel's law False", action="store_true")
        self.parser.add_argument('-d', '--dir', type=str, help="Working Directory", default="")
        self.parser.add_argument('-m', '--multi', help='Separate datasets scaled together', action="store_true")
        self.parser.add_argument('-1', '--single', action='store_true', default=False, help="Stop after a single step")
        self.parser.add_argument('--step', type=str, default=step, help="Step slug to start or resume from")
        self.parser.add_argument('--list-steps', action='store_true', default=False, help="List steps for the selected engine and exit")
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

        # engine selection (None when omitted so resume can auto-detect engine from latest checkpoint)
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument("--xds", action="store_const", const="XDS", dest="use", help="Use XDS Engine")
        group.add_argument("--dials", action="store_const", const="DIALS", dest="use", help="Use DIALS Engine")
        group.add_argument("--rugnux", action="store_const", const="RUGNUX", dest="use", help="Use Rugnux Engine")
        self.parser.set_defaults(use=None)

        self.step = step

        self.parser.add_argument(
            '--cluster', type=parse_cluster, help='Cluster parameters: partition:user@hostname,nodes,cpus'
        )

    @staticmethod
    def resolve_engine_name(args: argparse.Namespace) -> str:
        """
        Determine engine name from explicit CLI flag, latest checkpoint on resume, or default XDS.
        """
        if args.use:
            return args.use.upper()
        if not args.images:
            detected = peek_checkpoint_engine(Path(args.dir).absolute())
            if detected:
                return detected.upper()
        return "XDS"

    @staticmethod
    def get_engine_class(name: str) -> type[Analysis] | None:
        """Load and return the Analysis subclass for an engine name without instantiating it."""
        engine_name = name.upper()
        class_name = f'{engine_name}Analysis'
        module_name = f'mxproc.engines.{engine_name.lower()}'
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as err:
            logger.error(f'Analysis engine {engine_name} not found!')
            logger.exception(err)
            return None

    @staticmethod
    def get_engine(args: argparse.Namespace) -> Analysis | None:
        """
        Determine and instantiate the Analysis engine class from parsed arguments.
        :param args: Parsed arguments
        :return: Analysis instance or None
        """
        name = Application.resolve_engine_name(args)
        cls = Application.get_engine_class(name)
        if cls is None:
            return None
        return cls(args)

    def run(self, single: bool = False, argv: Sequence[str] | None = None) -> int:
        """
        Parse arguments and run analysis application.
        :param single: Run only one step
        :param argv: Optional argument list (defaults to sys.argv)
        """
        log.log_to_console(logging.DEBUG)
        args = self.parser.parse_args(argv)
        if args.list_steps:
            name = self.resolve_engine_name(args)
            cls = self.get_engine_class(name)
            if cls is None:
                return 1
            print(cls.format_steps())
            return 0

        proc = self.get_engine(args)
        if proc is None:
            return 1
        try:
            return proc.run(step=args.step, single=(single or args.single))
        except InvalidAnalysisStep as err:
            logger.error(str(err))
            return 1
