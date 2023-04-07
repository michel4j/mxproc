import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from pathlib import Path
from typing import Tuple, Sequence, Any, List, Dict

import numpy as np
import yaml
from mxio import parser
from scipy.integrate import nquad

from mxproc.log import logger
from mxproc.xtal import Lattice


class MissingLexicon(Exception):
    ...


class FilesMissing(Exception):
    ...


class InvalidAnalysisStep(Exception):
    ...


class Flag(IntFlag):

    def values(self) -> Tuple:
        return tuple(problem for problem in self.__class__ if problem in self)


class TextParser:
    LEXICON: dict

    @classmethod
    def parse(cls, filename: str, silent=False) -> dict:
        """
        Parse the provided file and return a dictionary
        :param filename: text file to parse
        :param silent: return empty dictionary instead of throwing exceptions
        """

        try:
            lex = cls.get_lexicon(Path(filename).name)
            info = parser.parse_file(filename, lex)
        except (MissingLexicon, FileNotFoundError, KeyError):
            info = {}
            if not silent:
                raise
        return info

    @classmethod
    def parse_text(cls, text: str, lexicon: dict) -> Any:
        """
        Parse the given text using the lexicon dictionary
        :param text: text to parse
        :param lexicon: lexicon specification dictionary
        """
        return parser.parse_text(lexicon, text)

    @classmethod
    def get_lexicon(cls, filename) -> dict:
        """
        Return the lexicon specified for a given file
        :param filename:
        :return: dictionary
        """

        try:
            spec_file = cls.LEXICON[filename]
            with open(spec_file, 'r') as file:
                specs = yaml.safe_load(file)
        except (KeyError, FileNotFoundError):
            raise MissingLexicon(f"No Lexicon available for file {filename!r}")

        return specs["root"]


class Workflow(IntEnum):
    SCREEN = 1
    PROCESS = 2

    def desc(self):
        return WORKFLOW_DESCRIPTIONS[self]


WORKFLOW_DESCRIPTIONS = {
    Workflow.SCREEN: 'Data Acquisition Strategy',
    Workflow.PROCESS: 'Data Processing',
}


class StepType(IntEnum):
    INITIALIZE = 0
    SPOTS = 1
    INDEX = 2
    STRATEGY = 3
    INTEGRATE = 4
    SYMMETRY = 5
    SCALE = 6
    EXPORT = 7
    REPORT = 8

    def prev(self):
        return StepType(max(min(StepType), self - 1))

    def next(self):
        return StepType(min(max(StepType), self + 1))

    def slug(self):
        return self.name.lower()

    def desc(self):
        return STEP_DESCRIPTIONS[self]


STEP_DESCRIPTIONS = {
    StepType.INITIALIZE: 'Initialization',
    StepType.SPOTS: 'Spot Search',
    StepType.INDEX: 'Auto-Indexing & Refinement',
    StepType.STRATEGY: 'Determining Optimal Strategy',
    StepType.INTEGRATE: 'Integration of Intensities',
    StepType.SYMMETRY: 'Determining & Applying Symmetry',
    StepType.SCALE: 'Scaling Intensities',
    StepType.EXPORT: 'Data Export',
    StepType.REPORT: 'Reports'
}


class StateType(IntFlag):
    SUCCESS = 0
    WARNING = 1
    FAILURE = 2


@dataclass
class Result:
    state: StateType = StateType.SUCCESS
    flags: int = 0
    messages: Sequence[str] = ()
    details: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtain and return a field from the details dictionary using dot notation, return the default if not found.
        :param key: field specification using dot separator notation
        :param default: default value if field is not found
        """
        return dict_field(self.details, key, default)


def dict_field(d: dict, key: str, default=None) -> Any:
    """
    Obtain and return a field from a dictionary using dot notation, return the default if not found.
    :param d: target dictionary
    :param key: field specification using dot separator notation
    :param default: default value if field is not found
    """

    if key in d:
        return d[key]
    elif "." in key:
        first, rest = key.split(".", 1)
        if first in d and isinstance(d[first], dict):
            return dict_field(d[first], rest, default)
    return default


def backup_files(*files: str):
    """
    Create numbered backups of specified files
    :param files: File names, relative or absolute paths
    """
    for filename in files:
        if os.path.exists(filename):
            index = 0
            while os.path.exists('%s.%0d' % (filename, index)):
                index += 1
            shutil.copy(filename, '%s.%0d' % (filename, index))


def generate_failure(message: str) -> Result:
    """
    Generate a Failure result
    :param message: failure message
    """
    messages = () if not message else (message,)

    return Result(
        state=StateType.FAILURE, flags=1, messages=messages, details={}
    )


class ResolutionMethod(Enum):
    EDGE = 0
    SIGMA = 1
    CC_HALF = 2
    R_FACTOR = 3
    MANUAL = 4


RESOLUTION_DESCRIPTION = {
    ResolutionMethod.EDGE: "detector edge",
    ResolutionMethod.SIGMA: "I/Sigma(I) > 1.0",
    ResolutionMethod.CC_HALF: "CC 1/2 Significance test",
    ResolutionMethod.R_FACTOR: "R-Factor < 30%",
    ResolutionMethod.MANUAL: "user request"
}


def select_resolution(table: List[dict], method: ResolutionMethod = ResolutionMethod.CC_HALF) -> Tuple[float, str]:
    """
    Takes a table of statistics and determines the optimal resolutions
    :param table: The table is a list of dictionaries each with at least the following fields shell, r_meas, cc_half
        i_sigma, signif
    :param method: Resolution Method
    :return: selected resolution, description of method used
    """

    data = np.array([
        (row['shell'], row['r_meas'], row['i_sigma'], row['cc_half'], int(bool(row['signif'].strip())))
        for row in table
    ], dtype=[
        ('shell', float),
        ('r_meas', float),
        ('i_sigma', float),
        ('cc_half', float),
        ('significance', bool)
    ])

    resolution = data['shell'][-1]
    used_method = ResolutionMethod.EDGE

    if method == ResolutionMethod.SIGMA:
        candidates = np.argwhere(data['i_sigma'] < 1.0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method
    elif method == ResolutionMethod.CC_HALF:
        candidates = np.argwhere(data['significance'] == 0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method
    elif method == ResolutionMethod.R_FACTOR:
        candidates = np.argwhere(data['r_meas'] > 30.0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method

    return resolution, RESOLUTION_DESCRIPTION[used_method]


def load_json(filename: Path | str) -> Any:
    """
    Load data from a JSON file
    :param filename: filename

    """
    with open(filename, 'r') as handle:
        info = json.load(handle)
    return info


def save_json(info: dict | list, filename: Path | str):
    """
    Save a list or dictionary into a JSON file

    :param info: data to save
    :param filename: json file

    """
    with open(filename, 'w') as handle:
        json.dump(info, handle)


def logistic(x, x0=0.0, weight=1.0, width=1.0, invert=False):
    mult = 1 if invert else -1
    return weight / (1 + np.exp(mult * width * (x - x0)))


@dataclass
class LogisticScore:
    name: str
    mean: float = 0
    weight: float = 1
    scale: float = 1

    def score(self, value: float) -> float:
        """
        Calculate and return a logistic score for a value
        :param value: target value
        """
        return self.weight / (1 + np.exp(self.scale * (self.mean - value)))


class ScoreManager:
    """
    A class which performs logistic scoring
    """
    items: Dict[str, LogisticScore]

    def __init__(self, specs: Dict[str, Tuple[float, float, float]]):
        """
        :param specs: A dictionary mapping item names to item parameters
        """

        total_weight = sum([weight for mean, weight, scale in specs.values()])
        self.items = {
            name: LogisticScore(name=name, mean=mean, weight=weight / total_weight, scale=scale)
            for name, (mean, weight, scale) in specs.items()
        }

    def score(self, **kwargs: float) -> float:
        """
        Calculate a score for the given values

        :param kwargs: key word arguments for values of items. missing values will get zero scores
        """

        return np.array([
            self.items[name].score(value) for name, value in kwargs.items() if name in self.items
        ]).sum()


def gaussian_fraction(r, fwhm_x, fwhm_y):
    def integrand(r, theta, sigma_x, sigma_y):
        return np.exp(
            -(r ** 2 * np.cos(theta) ** 2 / (2 * sigma_x ** 2) + r ** 2 * np.sin(theta) ** 2 / (2 * sigma_y ** 2))) * r

    sigma_x, sigma_y = fwhm_x / 2.355, fwhm_y / 2.355
    max_r = max(5 * sigma_x, 5 * sigma_y)
    total, _ = nquad(integrand, [(0, max_r), (0, 2 * np.pi)], args=(sigma_x, sigma_y))
    volume, _ = nquad(integrand, [(0, r), (0, 2 * np.pi)], args=(sigma_x, sigma_y))
    return volume / total


def show_warnings(label: str, messages: Sequence[str]):
    """
    Display a list of messages in the log
    :param label: Header label
    :param messages: sequence of messages

    """
    total = len(messages)
    if total:
        logger.warning(f'┬ {label}')
        for i, message in enumerate(messages):
            sym = '├' if i < total - 1 else '└'
            logger.warning(f'{sym} {message}')


def find_lattice(lattice: Lattice, candidates: Sequence[dict]) -> Tuple[Lattice, tuple]:
    """
    Given a lattice, generate a lattice and reindex matrix, from the compatible spacegroup candidates
    :param lattice: search lattice
    :param candidates: candidates from XDS CORRECT
    :return: lattice, reindex_matrix
    """

    for candidate in candidates:
        if lattice.character == candidate['character']:
            a, b, c, alpha, beta, gamma = candidate['unit_cell']
            new_lattice = Lattice(spacegroup=lattice.spacegroup, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            return new_lattice, candidate['reindex_matrix']

    return None, None
