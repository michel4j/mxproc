from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Sequence


@dataclass
class Experiment:
    name: str
    identifier: str
    reference: str
    directory: Path
    format: str
    detector: str
    wavelength: float
    frames: Sequence[Tuple[int, int]]
    template: str
    delta_angle: float
    start_angle: float = 0.0



