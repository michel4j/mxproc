import gzip
import subprocess
import asyncio
import time

from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, Sequence, Union, Literal

import numpy
import yaml

from mxio import DataSet, XYPair, parser
from mxproc import log
from numpy.typing import ArrayLike
from tqdm import tqdm

__all__ = [
    "logger",
    "SPACEGROUP_NAMES",
    "Experiment",
    "Analysis",
    "AnalysisOptions",
    "Command",
    "CommandFailed",
    "CommandNotFound",
    "run_command",
    "Lattice",
    "TextParser",
    "MissingLexicon",
]


logger = log.get_module_logger(__name__)

SPACEGROUP_NAMES = {
    1: 'P1', 2: 'P-1', 3: 'P2', 4: 'P2₁', 5: 'C2', 6: 'Pm', 7: 'Pc', 8: 'Cm', 9: 'Cc', 10: 'P2/m', 11: 'P2₁/m',
    12: 'C2/m', 13: 'P2/c', 14: 'P2₁/c', 15: 'C2/c', 16: 'P222', 17: 'P222₁', 18: 'P2₁2₁2', 19: 'P2₁2₁2₁',
    20: 'C222₁', 21: 'C222', 22: 'F222', 23: 'I222', 24: 'I2₁2₁2₁', 25: 'Pmm2', 26: 'Pmc2₁', 27: 'Pcc2', 28: 'Pma2',
    29: 'Pca2₁', 30: 'Pnc2', 31: 'Pmn2₁', 32: 'Pba2', 33: 'Pna2₁', 34: 'Pnn2', 35: 'Cmm2', 36: 'Cmc2₁', 37: 'Ccc2',
    38: 'Amm2', 39: 'Abm2', 40: 'Ama2', 41: 'Aba2', 42: 'Fmm2', 43: 'Fdd2', 44: 'Imm2', 45: 'Iba2', 46: 'Ima2',
    47: 'Pmmm', 48: 'Pnnn', 49: 'Pccm', 50: 'Pban', 51: 'Pmma', 52: 'Pnna', 53: 'Pmna', 54: 'Pcca', 55: 'Pbam',
    56: 'Pccn', 57: 'Pbcm', 58: 'Pnnm', 59: 'Pmmn', 60: 'Pbcn', 61: 'Pbca', 62: 'Pnma', 63: 'Cmcm', 64: 'Cmca',
    65: 'Cmmm', 66: 'Cccm', 67: 'Cmma', 68: 'Ccca', 69: 'Fmmm', 70: 'Fddd', 71: 'Immm', 72: 'Ibam', 73: 'Ibca',
    74: 'Imma', 75: 'P4', 76: 'P4₁', 77: 'P4₂', 78: 'P4₃', 79: 'I4', 80: 'I4₁', 81: 'P-4', 82: 'I-4', 83: 'P4/m',
    84: 'P4₂/m', 85: 'P4/n', 86: 'P4₂/n', 87: 'I4/m', 88: 'I4₁/a', 89: 'P422', 90: 'P42₁2', 91: 'P4₁22', 92: 'P4₁2₁2',
    93: 'P4₂22', 94: 'P4₂2₁2', 95: 'P4₃22', 96: 'P4₃2₁2', 97: 'I422', 98: 'I4₁22', 99: 'P4mm', 100: 'P4bm',
    101: 'P4₂cm', 102: 'P4₂nm', 103: 'P4cc', 104: 'P4nc', 105: 'P4₂mc', 106: 'P4₂bc', 107: 'I4mm', 108: 'I4cm',
    109: 'I4₁md', 110: 'I4₁cd', 111: 'P-42m', 112: 'P-42c', 113: 'P-42₁m', 114: 'P-42₁c', 115: 'P-4m2', 116: 'P-4c2',
    117: 'P-4b2', 118: 'P-4n2', 119: 'I-4m2', 120: 'I-4c2', 121: 'I-42m', 122: 'I-42d', 123: 'P4/mmm', 124: 'P4/mcc',
    125: 'P4/nbm', 126: 'P4/nnc', 127: 'P4/mbm', 128: 'P4/mnc', 129: 'P4/nmm', 130: 'P4/ncc', 131: 'P4₂/mmc',
    132: 'P4₂/mcm', 133: 'P4₂/nbc', 134: 'P4₂/nnm', 135: 'P4₂/mbc', 136: 'P4₂/mnm', 137: 'P4₂/nmc', 138: 'P4₂/ncm',
    139: 'I4/mmm', 140: 'I4/mcm', 141: 'I4₁/amd', 142: 'I4₁/acd', 143: 'P3', 144: 'P3₁', 145: 'P3₂', 146: 'R3',
    147: 'P-3', 148: 'R-3', 149: 'P312', 150: 'P321', 151: 'P3₁12', 152: 'P3₁21', 153: 'P3₂12', 154: 'P3₂21',
    155: 'R32', 156: 'P3m1', 157: 'P31m', 158: 'P3c1', 159: 'P31c', 160: 'R3m', 161: 'R3c', 162: 'P-31m', 163: 'P-31c',
    164: 'P-3m1', 165: 'P-3c1', 166: 'R-3m', 167: 'R-3c', 168: 'P6', 169: 'P6₁', 170: 'P6₅', 171: 'P6₂', 172: 'P6₄',
    173: 'P6₃', 174: 'P-6', 175: 'P6/m', 176: 'P6₃/m', 177: 'P622', 178: 'P6₁22', 179: 'P6₅22', 180: 'P6₂22',
    181: 'P6₄22', 182: 'P6₃22', 183: 'P6mm', 184: 'P6cc', 185: 'P6₃cm', 186: 'P6₃mc', 187: 'P-6m2', 188: 'P-6c2',
    189: 'P-62m', 190: 'P-62c', 191: 'P6/mmm', 192: 'P6/mcc', 193: 'P6₃/mcm', 194: 'P6₃/mmc', 195: 'P23', 196: 'F23',
    197: 'I23', 198: 'P2₁3', 199: 'I2₁3', 200: 'Pm-3', 201: 'Pn-3', 202: 'Fm-3', 203: 'Fd-3', 204: 'Im-3', 205: 'Pa-3',
    206: 'Ia-3', 207: 'P432', 208: 'P4₂32', 209: 'F432', 210: 'F4₁32', 211: 'I432', 212: 'P4₃32', 213: 'P4₁32',
    214: 'I4₁32', 215: 'P-43m', 216: 'F-43m', 217: 'I-43m', 218: 'P-43n', 219: 'F-43c', 220: 'I-43d', 221: 'Pm-3m',
    222: 'Pn-3n', 223: 'Pm-3n'
}


@dataclass
class Lattice:
    spacegroup: int = 0
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    def name(self):
        return f'{SPACEGROUP_NAMES[self.spacegroup]}'


@dataclass
class Experiment:
    name: str
    identifier: str
    reference: str
    directory: Path
    format: str
    detector: str
    wavelength: float
    distance: float
    frames: Sequence[Tuple[int, int]]
    template: str
    glob: str
    two_theta: float
    delta_angle: float
    pixel_size: XYPair
    detector_size: XYPair
    detector_origin: XYPair
    cutoff_value: float
    sensor_thickness: float
    start_angle: float = 0.0


@dataclass
class AnalysisOptions:
    files: Sequence[str]
    directory: Path
    working_directories: dict = field(default_factory=dict)
    anomalous: bool = False
    merge: bool = False


StepType = Literal["initialize", "spots", "index", "integrate", "scale", "symmetry", "files"]


class Analysis(ABC):
    experiments: Sequence[Experiment]
    options: AnalysisOptions
    results: dict   # results for each experiment keyed by experiment identifier

    def __init__(self, *files, directory: Union[Path, str] = "", anomalous: bool = False, merge: bool = False):
        self.options = AnalysisOptions(files=files, directory=Path(directory), anomalous=anomalous, merge=merge)
        self.experiments = load_multiple(files)
        self.results = {}

    def load(self, meta_file: Union[str, Path]):
        """
        Load an Analysis from a meta file and reset the state to it
        :param meta_file: Meta file to load
        """

        with gzip.open(meta_file, 'rb') as handle:  # gzip compressed yaml file
            meta = yaml.load(handle, yaml.Loader)

        self.options = AnalysisOptions(**meta['options'])
        self.experiments = [
            Experiment(**expt) for expt in meta['experiments']
        ]
        self.results = meta['results']

    def save(self, meta_file: Union[str, Path]):
        """
        Save analysis
        :param meta_file: file to save
        """
        meta = {
            'options': asdict(self.options),
            'experiments': [asdict(expt) for expt in self.experiments],
            'results': self.results
        }
        with gzip.open(meta_file, 'wb') as handle:  # gzip compressed yaml file
            yaml.dump(meta, handle, encoding='utf-8')

    def update_result(self, results: dict, realm: StepType):
        """
        Update the results dictionary and save a checkpoint meta file

        :param results: dictionary of results keyed by the experiment identifier
        :param realm: one of "initialize", "spots", "index", "integrate", "scale", "symmetry", "files"
        """

        for identifier, result in results.items():
            experiment_results = self.results.get(identifier, {})
            experiment_results.update({realm: result})
            self.results[identifier] = experiment_results

        meta_file = self.options.directory / f'{realm}.meta'
        self.save(meta_file)

    def run(self):
        """
        Perform the analysis and gather the harvested results
        """
        self.initialize()
        self.update_result(self.find_spots(), "spots")
        self.update_result(self.index(), "index")
        self.update_result(self.integrate(), "integrate")


    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the analysis, should be ready for spot finding after this.
        :param kwargs: keyword argument to tweak initialization
        """
        ...

    @abstractmethod
    def find_spots(self, **kwargs) -> dict:
        """
        Find spots, and prepare for indexing
        :param kwargs: keyword argument to tweak spot search
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def index(self, **kwargs) -> dict:
        """
        Perform indexing and refinement
        :param kwargs: keyword argument to tweak indexing
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    @abstractmethod
    def integrate(self, **kwargs) -> dict:
        """
        Perform integration.
        :param kwargs: keyword arguments for tweaking the integration settings.
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    #@abstractmethod
    def symmetry(self, **kwargs) -> dict:
        """
        Determination of Laue group symmetry and reindexing to the selected symmetry
        :param kwargs: keyword arguments for tweaking the symmetry
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    #@abstractmethod
    def scale(self, **kwargs) -> dict:
        """
        performs scaling on integrated datasets
        :param kwargs: keyword arguments for tweaking the scaling
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    #@abstractmethod
    def export(self, **kwargs) -> dict:
        """
        Export the results of processing into various formats.
        :param kwargs: keyword arguments for tweaking the export
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    #@abstractmethod
    def report(self, **kwargs) -> dict:
        """
        Generate reports of the analysis in TXT and HTML formats.
        :param kwargs: keyword arguments for tweaking the reporting
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...

    #@abstractmethod
    def quality(self, **kwargs) -> dict:
        """
        Check data quality.
        :param kwargs: keyword arguments for tweaking quality check
        :return: a dictionary, the keys are experiment identifiers and values are dictionaries containing
        harvested results from each dataset.
        """
        ...


class MissingLexicon(Exception):
    pass


class TextParser:
    LEXICON: dict

    @classmethod
    def parse(cls, filename: str) -> dict:
        try:
            spec_file = cls.LEXICON.get(filename)
            with open(spec_file, 'r') as file:
                specs = yaml.safe_load(file)
        except (KeyError, FileNotFoundError):
            raise MissingLexicon(f"No Lexicon available for file {filename!r}")

        info = parser.parse_file(filename, specs["root"])
        return info


def compress_series(values: ArrayLike) -> Sequence[Tuple[int, int]]:
    """
    Takes a sequence of integers such as [1,2,3,4,6,7,8,10] and compress it as a list of
    contiguous tuples [(1,4),(6,8), (10,10)]"

    :param values: ArrayLike
    :return: Sequence of Tuples.
    """

    values = numpy.array(values).astype(int)
    values.sort()
    return [
        (int(chunk[0]), int(chunk[-1]))
        for chunk in numpy.split(values, numpy.where(numpy.diff(values) > 1)[0] + 1)
        if len(chunk)
    ]


def load_experiment(filename: Union[str, Path]) -> Experiment:
    """
    Load an experiment from a dataset image file and return an Experiment instance

    :param filename: Full path to dataset image to import
    :return:/home/michel/Projects/mxproc/mxproc/importer.py
    """

    dset = DataSet.new_from_file(filename)
    if dset.index != dset.series[0]:
        dset.get_frame(index=dset.series[0])        # set to first frame so we get proper start angle

    if dset.frame.format in ["HDF5", "NXmx"]:
        wildcard = dset.reference
    else:
        wildcard = dset.glob

    return Experiment(
        name=dset.name,
        identifier=dset.identifier,
        reference=dset.reference,
        directory=dset.directory,
        format=dset.frame.format,
        detector=dset.frame.detector,
        wavelength=dset.frame.wavelength,
        distance=dset.frame.distance,
        frames=compress_series(dset.series),
        template=dset.glob,
        two_theta=dset.frame.two_theta,
        cutoff_value=dset.frame.cutoff_value,
        sensor_thickness=dset.frame.sensor_thickness,
        pixel_size=dset.frame.pixel_size,
        detector_size=dset.frame.size,
        detector_origin=dset.frame.center,
        delta_angle=dset.frame.delta_angle,
        start_angle=dset.frame.start_angle,
        glob=wildcard
    )


def load_multiple(file_names: Sequence[Union[str, Path]]) -> Sequence[Experiment]:
    """
    Load experiments from the provided file names and return a unique sequence of experiment objects representing them
    :param file_names:
    :return: Sequence of Experiments
    """
    experiments = {}
    for filename in file_names:
        experiment = load_experiment(filename)
        experiments[experiment.identifier] = experiment

    return list(experiments.values())


class CommandNotFound(Exception):
    ...


class CommandFailed(Exception):
    ...


class Command:
    def __init__(self, *args: str, logfile: Union[str, Path] = "commands.log", desc: str = ""):
        """
        Objects for running commands

        :param args: command arguments
        :param logfile: destination of standard output including errors
        :param desc: descriptive label of command
        """
        self.outfile = Path(logfile)
        self.args = " ".join(args)
        self.label = desc

    async def exec(self):
        """
        Main method to run the command asynchronously and update the progress bar with a descriptive label
        """

        with open(self.outfile, 'a') as stdout:
            start_time = time.time()
            start_str = datetime.now().strftime('%H:%M:%S')
            bar_fmt = "{desc}{elapsed}{postfix}"
            with tqdm(desc=f"{start_str} - {self.label} ... ", miniters=1, leave=False, bar_format=bar_fmt) as spinner:
                proc = await asyncio.create_subprocess_shell(self.args, stdout=stdout, stderr=stdout)
                while proc.returncode is None:
                    spinner.update()
                    await asyncio.sleep(.1)
            elapsed = time.time() - start_time

            if proc.returncode != 0:
                logger.error(log.log_value(f"- {self.label}", f"{elapsed:0.0f}s", log.TermColor.bold))
                raise subprocess.CalledProcessError(proc.returncode, self.args)
            else:
                logger.info(log.log_value(f"- {self.label}", f"{elapsed:0.0f}s", log.TermColor.bold))

    def run(self):
        """
        Run command in an event loop
        :return:
        """
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.exec())
        except subprocess.CalledProcessError as err:
            raise CommandFailed(f"{err}")


def run_command(*args, desc: str = "", logfile: Union[str, Path] = "commands.log"):
    """
    Creates and executes a command instance

    :param args: command arguments
    :param logfile: destination of standard output including errors
    :param desc: descriptive label of command
    """

    command = Command(*args, desc=desc, logfile=logfile)
    command.run()

