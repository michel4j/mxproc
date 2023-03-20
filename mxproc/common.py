import os
import shutil
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering, reduce
from typing import Tuple, Sequence, Any, List

import yaml
import numpy
from mxio import parser


class MissingLexicon(Exception):
    ...


class FilesMissing(Exception):
    ...


class InvalidAnalysisStep(Exception):
    ...


class Flag(Enum):
    def matches(self, flags: int) -> bool:
        return bool(self.value & flags)

    @classmethod
    def flags(cls, code: int) -> Tuple:
        return tuple(problem for problem in cls if problem.value & code)

    @staticmethod
    def code(*flags: Sequence[Enum]):
        return reduce(lambda x, y: x.value | y.value, flags)

    @staticmethod
    def has(code: int, test: int) -> bool:
        """
        Test multiple bits to determine whether they are on irrespective of other bits.
        :param code: full integer containing multiple bits
        :param test: integer containing only the bits to check in code
        :return: bool, True if at least the specified bits are on
        """
        return code | test == test

    @staticmethod
    def has_only(code: int, test: int) -> bool:
        """
        Test multiple bits to determine whether they are on and all other bits are off.
        :param code: full integer containing multiple bits
        :param test: integer containing only the bits to check in code
        :return: bool, True if only the specified bits are on
        """
        return code & test == test

    def __lt__(self, other: 'Flag'):
        return other.value > self.value


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


@total_ordering
class AnalysisStep(Enum):
    INITIALIZE = 0
    SPOTS = 1
    INDEX = 2
    INTEGRATE = 3
    SYMMETRY = 4
    SCALE = 5
    QUALITY = 6
    EXPORT = 7
    REPORT = 8

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def slug(self):
        return self.name.lower()

    def desc(self):
        return STEP_DESCRIPTIONS[self]


STEP_DESCRIPTIONS = {
    AnalysisStep.INITIALIZE: 'Initialization',
    AnalysisStep.SPOTS: 'Spot Search',
    AnalysisStep.INDEX: 'Auto-Indexing & Refinement',
    AnalysisStep.INTEGRATE: 'Integration of Intensities',
    AnalysisStep.SYMMETRY: 'Determining & Applying Symmetry',
    AnalysisStep.SCALE: 'Scaling Intensities',
    AnalysisStep.QUALITY: 'Data Quality Evaluation',
    AnalysisStep.EXPORT: 'Data Export',
    AnalysisStep.REPORT: 'Reports'
}


class StateType(Flag):
    SUCCESS = 0
    WARNING = 1
    FAILURE = 2


@dataclass
class Status:
    state: StateType
    messages: Sequence[str] = ()
    flags: int = 0


@dataclass
class Result:
    status: Status
    details: Any


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
        status=Status(StateType.FAILURE, flags=1, messages=messages),  details={}
    )


class ResolutionMethod(Enum):
    EDGE = 0
    SIGMA = 1
    CC_HALF = 2
    R_FACTOR = 3
    MANUAL = 4


RESOLUTION_DESCRIPTION = {
    ResolutionMethod.EDGE:  "detector edge",
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

    data = numpy.array([
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
        candidates = numpy.argwhere(data['i_sigma'] < 1.0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method
    elif method == ResolutionMethod.CC_HALF:
        candidates = numpy.argwhere(data['significance'] == 0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method
    elif method == ResolutionMethod.R_FACTOR:
        candidates = numpy.argwhere(data['r_meas'] > 30.0).ravel()
        if len(candidates):
            resolution = data['shell'][candidates[0]]
            used_method = method

    return resolution, RESOLUTION_DESCRIPTION[used_method]
