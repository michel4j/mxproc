import numpy
from numpy.typing import ArrayLike
from typing import Union, Sequence, Tuple
from pathlib import Path

from mxio import DataSet
from mxproc import Experiment


def summarize_list(values: ArrayLike) -> Sequence[Tuple[int, int]]:
    """
    Takes a list of integers such as [1,2,3,4,6,7,8,10] and summarises it as a list of
    contiguous tuples [(1,4),(6,8), (10,10)]"

    :param values: ArrayLike
    :return: Sequence of Tuples.
    """

    values = numpy.array(values)
    values.sort()
    return [
        (chunk[0], chunk[-1])
        for chunk in numpy.split(values, numpy.where(numpy.diff(values) > 1)[0] + 1)
        if len(chunk)
    ]


def load_experiment(filename: Union[str, Path]) -> Experiment:
    """
    Load an experiment from a dataset image file and return an Experiment instance

    :param filename: Full path to dataset image to import
    :return:
    """

    dset = DataSet.new_from_file(filename)
    dset.get_frame(index=dset.series[0])        # set to first frame so we get proper start angle

    return Experiment(
        name=dset.name,
        identifier=dset.identifier,
        reference=dset.reference,
        directory=dset.directory,
        format=dset.frame.format,
        detector=dset.frame.detector,
        wavelength=dset.frame.wavelength,
        frames=summarize_list(dset.series),
        template=dset.glob,
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

