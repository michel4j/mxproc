from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, Union

import itertools
import numpy
from mxio import DataSet, XYPair, Geometry
from numpy.typing import ArrayLike

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

MATTHEWS_COEFFICIENT = 2.69     # From Protein Sci. 2003 Sep; 12(9): 1865–1871.
SPACEGROUPS = {
    'aP': (1, 2),
    'mP': (3, 4, 6, 7, 10, 11, 13, 14),
    'mC': (5, 8, 9, 12, 15),
    'mI': (5, 8, 9, 12, 15),
    'oP': (
        16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 69, 61, 62,
    ),
    'oC': (20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 68,),
    'oF': (22, 38, 39, 40, 41, 42, 43, 69, 70,),
    'oI': (23, 24, 44, 45, 46, 71, 72, 73, 74, ),
    'tP': (
        75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103,
        104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 136, 137, 138,
    ),
    'tI': (79, 97, 79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142),
    'hP': (
        143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165,
        168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
        186, 187, 188, 189, 190, 191, 192, 193, 194
    ),
    'hR': (146, 148, 155, 160, 161, 166, 167),
    'cP': (195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, 221, 222, 223, 224),
    'cF': (196, 202, 203, 209, 210, 216, 225, 226, 227, 228,),
    'cI': (197, 199, 204, 206, 211, 214, 217, 220, 229, 230)
}

POINT_GROUPS = {
    'aP': (1,),
    'mP': (3,),
    'mC': (5,),
    'mI': (5,),
    'oP': (16,),
    'oC': (21,),
    'oF': (22,),
    'oI': (23,),
    'tP': (75, 89),
    'tI': (79, 97),
    'hP': (143, 149, 150, 168, 177),
    'hR': (146, 155),
    'cP': (195, 207),
    'cF': (196, 209),
    'cI': (197, 211,)
}


@dataclass
class Lattice:
    spacegroup: int = 0
    name: str = field(init=False)
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    character: str = field(init=False)

    def __post_init__(self):
        self.character = spacegroup_character(self.spacegroup)
        self.name = SPACEGROUP_NAMES.get(self.spacegroup, "P1")

    def cell_text(self):
        return f"{self.a:0.3f} {self.b:0.3f} {self.c:0.3f} {self.alpha:0.2f} {self.beta:0.2f} {self.gamma:0.2f}"

    def volume(self) -> float:
        """
        Calculate unit cell volume in  cubic angstroms
        """

        cos_alpha = numpy.cos(numpy.radians(self.alpha))
        cos_beta = numpy.cos(numpy.radians(self.beta))
        cos_gamma = numpy.cos(numpy.radians(self.gamma))
        return self.a * self.b * self.c * (
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma
        )**0.5

    def estimate_content(self) -> int:
        """
        Estimate the number of amino acids in the unit cell assuming 50% solvent content
        and a Matthews coefficient of 2.69 A^3/Dalton, 110 Daton per Amino acid
        """
        return int(self.volume()/(MATTHEWS_COEFFICIENT * 110))


@dataclass
class Beam:
    flux: float
    fwhm_x: float
    fwhm_y: float
    aperture: float
    wavelength: float = 1.0


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
    geometry: Geometry
    cutoff_value: float
    sensor_thickness: float
    start_angle: float = 0.0
    lattice: Lattice = field(default_factory=Lattice)
    missing: Sequence[Tuple[int, int]] = ()


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


def load_experiment(filename: Union[str, Path]) -> Sequence[Experiment]:
    """
    Load an experiment from a dataset image file and return one or more Experiment instances

    :param filename: Full path to dataset image to import
    """

    dset = DataSet.new_from_file(filename)
    if dset.index != dset.series[0]:
        dset.get_frame(index=dset.series[0])  # set to first frame so we get proper start angle

    if dset.frame.format in ["HDF5", "NXmx"]:
        wildcard = dset.reference
        angles = numpy.array([
            (index, dset.frame.start_angle + (index - 1) * dset.frame.delta_angle)
            for index in dset.series
        ])
    else:
        wildcard = dset.glob
        angles = numpy.array([
            (index, frame.start_angle)
            for index in dset.series
            for frame in (dset.get_frame(index),)
        ])

    diffs = numpy.abs(numpy.diff(angles, axis=0))
    diffs[:, 1] /= dset.frame.delta_angle
    diffs = numpy.round(diffs)

    sweep_mask = numpy.argwhere(diffs[:, 0] != diffs[:, 1]).ravel() + 1
    sweeps = numpy.split(angles, sweep_mask)

    add_suffix = len(sweeps) > 1
    return [
        Experiment(
            name=f"{dset.name}" + (f"-{i + 1}" if add_suffix else ""),
            identifier=f"{dset.identifier}-{i + 1}",
            reference=dset.template.format(field=sweep[0, 0]),
            directory=dset.directory,
            format=dset.frame.format,
            detector=dset.frame.detector,
            wavelength=dset.frame.wavelength,
            distance=dset.frame.distance,
            frames=compress_series(sweep[:, 0]),
            template=dset.glob,
            two_theta=dset.frame.two_theta,
            cutoff_value=dset.frame.cutoff_value,
            sensor_thickness=dset.frame.sensor_thickness,
            pixel_size=dset.frame.pixel_size,
            detector_size=dset.frame.size,
            detector_origin=dset.frame.center,
            geometry=dset.frame.geometry,
            delta_angle=dset.frame.delta_angle,
            start_angle=sweep[0, 1],
            glob=wildcard,
            missing=compress_series(
                numpy.setdiff1d(
                    numpy.arange(sweep[0, 0], sweep[-1, 0] + 1),
                    sweep[:, 0], assume_unique=True
                )
            )
        )
        for i, sweep in enumerate(sweeps)
    ]


def load_multiple(file_names: Sequence[Union[str, Path]]) -> Sequence[Experiment]:
    """
    Load experiments from the provided file names and return a unique sequence of experiment objects representing them
    :param file_names:
    :return: Sequence of Experiments
    """
    experiments = {}
    for filename in file_names:
        for experiment in load_experiment(filename):
            experiments[experiment.identifier] = experiment

    return list(experiments.values())


def lattice_point_groups(*characters: str) -> Sequence[str]:
    """
    Takes a list of lattice characters and returns a unique list of the
    names of the lowest symmetry pointgroup.

    :param characters: list of lattice character strings
    """

    point_groups = sorted(itertools.chain.from_iterable([
        POINT_GROUPS[character] for character in characters
    ]))

    # use a dictionary to reduce to unique keys, and keep sort order
    point_groups_dict = {
        SPACEGROUP_NAMES[point_group]: point_group
        for point_group in point_groups
    }
    return list(point_groups_dict.keys())


def spacegroup_character(number: int) -> str:
    """
    Determine and return the Bravais lattice character from the given spacegroup number
    :param number: space group number
    :return: 2-character string such as "aP", "mC", etc
    """

    for character, spacegroups in SPACEGROUPS.items():
        if number in spacegroups:
            return character

    return "aP"


def resolution_shells(resolution: float) -> ArrayLike:
    """
    Calculate resolution shells for the given d-spacing
    :param resolution: minimum d-spacing
    :return: array of resolution shells
    """

    count = int(20 - 2**(resolution - 2))
    if 6 - resolution > 1.0:
        return numpy.round(numpy.sqrt(1/(numpy.linspace(1.0/6**2, 1.0/resolution**2, count))), 2)
    else:
        return []
