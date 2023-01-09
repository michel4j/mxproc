import gzip
import subprocess
import sys

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, Sequence, Union

import numpy
import yaml
from mxio import DataSet, XYPair
from numpy.typing import ArrayLike
from tqdm import tqdm

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
        return f'{self.spacegroup}'


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


class Analysis(ABC):
    experiments: Sequence[Experiment]
    options: AnalysisOptions
    results: dict

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

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the analysis, should be ready for spot finding after this.
        :param kwargs: keyword argument to tweak initialization
        """
        ...

    #@abstractmethod
    def find_spots(self, **kwargs):
        """
        Find spots, and prepare for indexing
        :param kwargs: keyword argument to tweak spot search
        """
        ...

    #@abstractmethod
    def index(self, **kwargs):
        """
        Perform indexing and refinement
        :param kwargs: keyword argument to tweak indexing
        """
        ...

    #@abstractmethod
    def integrate(self, **kwargs):
        """
        Perform integration.
        :param kwargs: keyword arguments for tweaking the integration settings.
        """
        ...

    #@abstractmethod
    def symmetry(self, **kwargs):
        """
        Determination of Laue group symmetry and reindexing to the selected symmetry
        :param kwargs: keyword arguments for tweaking the symmetry
        """
        ...

    #@abstractmethod
    def scale(self, **kwargs):
        """
        performs scaling on integrated datasets
        :param kwargs: keyword arguments for tweaking the scaling
        """
        ...

    #@abstractmethod
    def export(self, **kwargs):
        """
        Export the results of processing into various formats.
        :param kwargs: keyword arguments for tweaking the export
        """
        ...

    #@abstractmethod
    def report(self, **kwargs):
        """
        Generate reports of the analysis in TXT and HTML formats.
        :param kwargs: keyword arguments for tweaking the reporting
        """
        ...

    #@abstractmethod
    def quality(self, **kwargs):
        """
        Check data quality.
        :param kwargs: keyword arguments for tweaking quality check
        """
        ...


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


def run_command(*args):
    try:
        # create a default tqdm progress bar object, unit='B' definnes a String that will be used to define the unit of each iteration in our case bytes
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="run_task={}".format(args)) as t:
            process = subprocess.Popen(args, shell=True, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            # print subprocess output line-by-line as soon as its stdout buffer is flushed in Python 3:
            for line in process.stdout:
                # Update the progress, since we do not have a predefined iterator
                # tqdm doesnt know before hand when to end and cant generate a progress bar
                # hence elapsed time will be shown, this is good enough as we know
                # something is in progress
                t.update()
                # forces stdout to "flush" the buffer
                sys.stdout.flush()

            process.stdout.close()

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, args)

    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            "common::run_command() : [ERROR]: output = %s, error code = %s\n"
            % (e.output, e.returncode))