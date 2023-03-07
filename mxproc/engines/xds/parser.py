import re
import numpy

from pathlib import Path
from typing import Any, Tuple
from mxproc.common import TextParser, Status, StateType, MissingLexicon, FilesMissing, Flag
from mxproc.log import logger
from mxproc.experiment import lattice_point_groups


DATA_PATH = Path(__file__).parent / "data"


class IndexProblem(Flag):
    NONE = 0
    COMMAND_ERROR = 1 << 0
    LOW_FRACTION  = 1 << 1
    FEW_SPOTS     = 1 << 2
    POOR_SOLUTION = 1 << 3
    REFINEMENT    = 1 << 4
    TERMINATED    = 1 << 5
    SUBTREES      = 1 << 6
    INDEX_ORIGIN  = 1 << 7
    LOW_DIMENSION = 1 << 8
    INDICES       = 1 << 9


INDEX_FAILURES = {
    r'CANNOT CONTINUE WITH A TWO-DIMENSIONAL': IndexProblem.LOW_DIMENSION,
    r'DIMENSION OF DIFFERENCE VECTOR SET LESS THAN \d+.': IndexProblem.LOW_DIMENSION,
    r'INSUFFICIENT NUMBER OF ACCEPTED SPOTS.': IndexProblem.FEW_SPOTS,
    r'SOLUTION IS INACCURATE': IndexProblem.POOR_SOLUTION,
    r'RETURN CODE IS IER= \s+\d+': IndexProblem.REFINEMENT,
    r'CANNOT INDEX REFLECTIONS': IndexProblem.TERMINATED,
    r'^INSUFFICIENT PERCENTAGE .+ OF INDEXED REFLECTIONS': IndexProblem.LOW_FRACTION,
    r'REFINEMENT DID NOT CONVERGE': IndexProblem.REFINEMENT,
    r'REDUCE returns IRANK= \s+.+': IndexProblem.TERMINATED,
}

INDEX_MESSAGES = {
    IndexProblem.LOW_DIMENSION: 'Cluster dimensions not 3D',
    IndexProblem.LOW_FRACTION: 'Many un-indexed spots',
    IndexProblem.FEW_SPOTS: 'Insufficient strong spots',
    IndexProblem.POOR_SOLUTION: 'Poor Solution',
    IndexProblem.REFINEMENT: 'Unable to refine solution',
    IndexProblem.TERMINATED: 'No indexing solution',
    IndexProblem.SUBTREES: 'Multiple lattices',
    IndexProblem.INDEX_ORIGIN: 'Sub-optimal index origin',
    IndexProblem.COMMAND_ERROR: 'Command failed to run'
}

INDEX_STATES = {
    IndexProblem.LOW_DIMENSION: StateType.FAILURE,
    IndexProblem.FEW_SPOTS: StateType.FAILURE,
    IndexProblem.POOR_SOLUTION: StateType.FAILURE,
    IndexProblem.REFINEMENT: StateType.FAILURE,
    IndexProblem.TERMINATED: StateType.FAILURE,
    IndexProblem.LOW_FRACTION: StateType.WARNING,
    IndexProblem.SUBTREES: StateType.WARNING,
    IndexProblem.INDEX_ORIGIN: StateType.WARNING,
    IndexProblem.COMMAND_ERROR: StateType.FAILURE,
    IndexProblem.INDICES: StateType.FAILURE,
}


def get_failure(message: str, failures) -> Any:
    """
    Search for matching failures in a dictionary of pattern, value pairs and return the corresponding
    Failure code or None if not found

    :param message: String
    :param failures: dictionary mapping match patterns to failure codes
    :return: failure code
    """
    if not message:
        return IndexProblem.NONE

    for pattern, code in failures.items():
        if re.match(pattern, message):
            return code


class XDSParser(TextParser):
    LEXICON = {
        "COLSPOT.LP": DATA_PATH / "spots.yml",
        "IDXREF.LP": DATA_PATH / "idxref.yml",
        "INTEGRATE.LP": DATA_PATH / "integrate.yml",
        "CORRECT.LP": DATA_PATH / "correct.yml",
        "XSCALE.LP": DATA_PATH / "xscale.yml",
        "XDSSTAT.LP": DATA_PATH / "xdsstat.yml",
        "XPARM.XDS": DATA_PATH / "xparm.yml",
        "GXPARM.XDS": DATA_PATH / "xparm.yml"
    }

    @classmethod
    def parse_index(cls):
        details = {}
        code = 0
        state = StateType.SUCCESS
        try:
            log_details = cls.parse('IDXREF.LP')
            for msg in [log_details.get('failure_message'), log_details.get('warning_message')]:
                failure = get_failure(msg, INDEX_FAILURES)
                code |= failure.value
                state = INDEX_STATES.get(failure)
            param_details = cls.parse('XPARM.XDS')

        except (FilesMissing, FileNotFoundError)as err:
            code |= IndexProblem.TERMINATED.value
        except MissingLexicon as err:
            code |= IndexProblem.TERMINATED.value
        else:
            details.update(param_details.get('parameters', {}))
            details['subtrees'] = [tree['population'] for tree in log_details.get('subtrees', [])]

            # check for multiple lattices as smaller subtree with at least 5% of spots in main subtree
            if len(details['subtrees']) > 1 and 100.0 * details['subtrees'][1]/details['subtrees'][0] > .1:
                code |= IndexProblem.SUBTREES.value

            details['overlaps'] = log_details.get('delta_overlaps', [])
            details['index_origins'] = log_details.get('index_origins', [])
            details['lattices'] = log_details.get('lattices', [])
            details['point_groups'] = lattice_point_groups([lattice['character'] for lattice in details['lattices']])
            details['spots'] = log_details.get('spots', {})

            indices = numpy.array([cluster['hkl'] for cluster in log_details.get('cluster_indices', [])])
            index_quality = numpy.abs(indices - indices.astype(int)).mean()
            details['quality'] = log_details.get('quality', {})
            details['quality'].update(indices=index_quality)

            if index_quality > log_details.get('max_integral_dev', 0.05):
                code |= IndexProblem.INDICES.value

        message = "; ".join(
            INDEX_MESSAGES.get(flag, '')
            for flag in IndexProblem.flags(code) if flag.value > 0
        )
        status = Status(state=state, message=message, flags=code)
        return {'status': status, 'details': details}

