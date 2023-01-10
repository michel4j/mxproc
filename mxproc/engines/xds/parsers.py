import yaml
from mxio import parser
from enum import Enum
from pathlib import Path


class IndexFailure(Enum):
    NONE = 0
    LOW_DIMENSION = 1
    LOW_FRACTION = 2
    FEW_SPOTS = 3
    POOR_SOLUTION = 4
    REFINEMENT = 4
    TERMINATED = 5


FAILURES = {
    r'CANNOT CONTINUE WITH A TWO-DIMENSIONAL': IndexFailure.LOW_DIMENSION,
    r'DIMENSION OF DIFFERENCE VECTOR SET LESS THAN \d+.': IndexFailure.LOW_DIMENSION,
    r'INSUFFICIENT NUMBER OF ACCEPTED SPOTS.': IndexFailure.FEW_SPOTS,
    r'SOLUTION IS INACCURATE': IndexFailure.POOR_SOLUTION,
    r'RETURN CODE IS IER= \s+\d+': IndexFailure.REFINEMENT,
    r'CANNOT INDEX REFLECTIONS': IndexFailure.TERMINATED,
    r'^INSUFFICIENT PERCENTAGE .+ OF INDEXED REFLECTIONS': IndexFailure.LOW_FRACTION,
}


class XDSParsers:

    @classmethod
    def find_spots(cls):
        spec_file = Path(__file__).parent / "data" / "spots.yml"
        with open(spec_file, 'r') as file:
            specs = yaml.safe_load(file)

        log_file = Path("COLSPOT.LP")
        info = parser.parse_file(log_file, specs["root"])
        return info


