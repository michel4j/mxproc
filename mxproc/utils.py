import json
import gzip

from pathlib import Path
from typing import Union


def load_meta(filename: Union[str, Path]) -> dict:
    """
    Load metadata from gzip compressed json file and return a dictionary
    :param filename: A string or a Path
    :return: Dictionary
    """

    with gzip.open(filename, 'rb') as handle:
        info = json.load(handle)
    return info


def save_meta(info: dict, filename: Union[Path, str]):
    """
    Save a dictionary as a gzip compressed json meta file
    :param info: dictionary to save
    :param filename: path to save the file as
    """

    with gzip.open(filename, 'wb') as handle:
        json.dump(info, handle)  # msgpack_numpy.encode

