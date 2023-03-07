from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Sequence
from functools import total_ordering, reduce

import yaml
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

    @classmethod
    def code(cls, *flags: Sequence[Enum]):
        return reduce(lambda x, y: x.value | y.value, flags)


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


class StateType(Enum):
    SUCCESS = 0
    WARNING = 1
    FAILURE = 2


@dataclass
class Status:
    state: StateType
    message: str
    flags: int = 0
