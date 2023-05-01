import re
import os
import numpy
from enum import auto
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Tuple
from mxproc.common import TextParser, StateType, MissingLexicon, FilesMissing, Flag, Result
from mxproc.log import logger
from mxproc.xtal import lattice_point_groups, Lattice


DATA_PATH = Path(__file__).parent / "data"
MAX_SUBTREE_RATIO = 0.05     # maximum fraction of reflections allowed in second subtree before flagging as multi-lattice


class IndexProblem(Flag):
    NONE = auto()
    SOFTWARE_ERROR = auto()
    INSUFFICIENT_SPOTS = auto()
    LOW_CLUSTER_DIMENSION = auto()
    NON_INTEGRAL_INDICES = auto()
    MULTIPLE_SUBTREES = auto()
    LOW_INDEXED_FRACTION = auto()
    POOR_SOLUTION = auto()
    REFINEMENT_FAILED = auto()
    INDEXING_FAILED = auto()
    INVERTED_AXIS = auto()
    FRACTIONAL_INDICES = auto()
    WRONG_SPOT_PARAMETERS = auto()


INDEX_FAILURES = {
    r'CANNOT CONTINUE WITH A TWO-DIMENSIONAL': IndexProblem.LOW_CLUSTER_DIMENSION,
    r'DIMENSION OF DIFFERENCE VECTOR SET LESS THAN \d+.': IndexProblem.LOW_CLUSTER_DIMENSION,
    r'INSUFFICIENT NUMBER OF ACCEPTED SPOTS.': IndexProblem.INSUFFICIENT_SPOTS,
    r'SOLUTION IS INACCURATE': IndexProblem.POOR_SOLUTION,
    r'RETURN CODE IS IER= \s+\d+': IndexProblem.INDEXING_FAILED,
    r'CANNOT INDEX REFLECTIONS': IndexProblem.INDEXING_FAILED,
    r'^INSUFFICIENT PERCENTAGE .+ OF INDEXED REFLECTIONS': IndexProblem.LOW_INDEXED_FRACTION,
    r'REFINEMENT DID NOT CONVERGE': IndexProblem.REFINEMENT_FAILED,
    r'REDUCE returns IRANK= \s+.+': IndexProblem.INDEXING_FAILED,
}

INDEX_STATES = {
    IndexProblem.SOFTWARE_ERROR: (StateType.FAILURE, 'Program failed'),
    IndexProblem.INSUFFICIENT_SPOTS: (StateType.WARNING, 'Insufficient number of strong spots'),
    IndexProblem.LOW_CLUSTER_DIMENSION: (StateType.WARNING, 'Cluster dimensions not 3D'),
    IndexProblem.NON_INTEGRAL_INDICES: (StateType.WARNING, 'Cluster indices deviate from integers'),
    IndexProblem.MULTIPLE_SUBTREES: (StateType.WARNING, 'Multiple lattices'),
    IndexProblem.LOW_INDEXED_FRACTION: (StateType.WARNING, 'Low fraction of indexed spots'),
    IndexProblem.POOR_SOLUTION: (StateType.WARNING, 'Indexing solution too poor'),
    IndexProblem.REFINEMENT_FAILED: (StateType.FAILURE, 'Failed to refine solution'),
    IndexProblem.INDEXING_FAILED: (StateType.FAILURE, 'Auto-Indexing failed for unknown reason'),
    IndexProblem.INVERTED_AXIS: (StateType.WARNING, 'Experimental Parameters may be poorly specified'),
    IndexProblem.FRACTIONAL_INDICES: (StateType.WARNING, 'Many half-integer cluster indices'),
    IndexProblem.WRONG_SPOT_PARAMETERS: (StateType.WARNING, 'Spot are closer than allowed')
}


def get_failure(message: str, failures) -> Any:
    """
    Search for matching failures in a dictionary of pattern, value pairs and return the corresponding
    Failure code or None if not found

    :param message: String
    :param failures: dictionary mapping match patterns to failure codes
    :return: failure code
    """
    if message:
        for pattern, problem in failures.items():
            if re.match(pattern, message):
                return problem

    return IndexProblem.NONE


def max_sub_range(values: NDArray, size: int) -> Tuple[int, int]:
    """
    Find the range of values of a given size which maximizes the sum of the sub array
    :param values: full array values
    :param size: sub-array size.
    :return: Tuple
    """
    max_index = numpy.argmax([values[i:i+size].sum() for i in range(len(values)-size)])
    return max_index, max_index + size


def get_spot_distribution() -> Any:
    """
    Calculate the percentage of indexed spots per image

    :return: 2D array mapping frame number to percentage of spots indexed.
    """

    data = numpy.loadtxt('SPOT.XDS', comments='!')
    if len(data.shape) < 2:
        return None, None

    spots = numpy.empty((data.shape[0], 4), dtype=numpy.uint16)
    spots[:, :3] = numpy.round(data[:, :3]).astype(numpy.uint16)

    if data.shape[1] > 4:
        spots[:, 3] = numpy.abs(data[:, 4:]).sum(axis=1) > 0
    else:
        spots[:, 3] = 0

    indexed = spots[:, 3] == 1

    total, edges = numpy.histogram(spots[:, 2], bins=50)
    indexed_counts, _ = numpy.histogram(spots[indexed, 2], bins=50)

    results = numpy.empty((50, 3), dtype=numpy.uint16)
    results[:, 0] = ((edges[1:] + edges[:-1])*.5).astype(int)
    results[:, 1] = total
    results[:, 2] = indexed_counts

    fraction_indexed = numpy.divide(
        indexed_counts, total, out=numpy.zeros(total.shape, dtype=numpy.float64), where=total != 0
    )
    range_start, range_end = max_sub_range(fraction_indexed, len(fraction_indexed)//4)

    return results, (results[range_start, 0], results[range_end, 0])


class XDSParser(TextParser):
    LEXICON = {
        "COLSPOT.LP": DATA_PATH / "spots.yml",
        "IDXREF.LP": DATA_PATH / "idxref.yml",
        "XPLAN.LP": DATA_PATH / "xplan.yml",
        "INTEGRATE.LP": DATA_PATH / "integrate.yml",
        "CORRECT.LP": DATA_PATH / "correct.yml",
        "XSCALE.LP": DATA_PATH / "xscale.yml",
        "XDSSTAT.LP": DATA_PATH / "xdsstat.yml",
        "XPARM.XDS": DATA_PATH / "xparm.yml",
        "GXPARM.XDS": DATA_PATH / "xparm.yml",
        "XTRIAGE.LP": DATA_PATH / "xtriage.yml"
    }

    @classmethod
    def parse_index(cls) -> Result:
        details = {}
        problems = IndexProblem.NONE
        try:
            log_details = cls.parse('IDXREF.LP')
            param_details = cls.parse('XPARM.XDS', silent=True)
        except (FilesMissing, FileNotFoundError, MissingLexicon) as err:
            logger.exception(err)
            problems |= IndexProblem.SOFTWARE_ERROR
        else:
            lattice = Lattice(param_details.pop('sg_number', 1), *param_details.pop('unit_cell', ()))
            details.update(param_details)
            details['lattice'] = lattice
            details['subtrees'] = [tree['population'] for tree in log_details.get('subtrees', [])]
            details['overlaps'] = log_details.get('delta_overlaps', [])
            details['index_origins'] = log_details.get('index_origins', [])
            details['lattices'] = log_details.get('lattices', [])
            details['point_groups'] = lattice_point_groups(*[lattice['character'] for lattice in details['lattices']])
            details['quality'] = log_details.get('quality', {})

            # spots distribution
            exp_ang_err = log_details.get('exp_ang_err', 2.0)
            exp_pos_err = log_details.get('exp_pos_err', 3.0)

            details['spots'] = log_details.get('spots', {})
            spot_counts, best_range = get_spot_distribution()
            details['spots'].update(counts=spot_counts, best_range=best_range)

            # large fraction of rejected spots far from ideal
            misfit_percent = 100 * details['spots'].get('misfits', 0)/details['spots'].get('total', 1)

            details['quality'].update(
                misfit_percent=misfit_percent,      # Percentage of spots too far from ideal position
                expected_angle_error=exp_ang_err,
                expected_pixel_error=exp_pos_err,
            )

            index_deviation = 0.0
            half_index_pct = 0.0
            max_deviation = log_details.get('max_integral_dev', 0.05)
            if 'cluster_indices' in log_details:
                indices = numpy.array([cluster['hkl'] for cluster in log_details.get('cluster_indices', [])])
                if len(indices):
                    fractional_indices = 0.5 * numpy.round(indices) * 2
                    integer_indices = numpy.round(indices)
                    index_deviation = numpy.abs(indices - integer_indices).mean()
                    half_index_pct = 100 * (numpy.abs(fractional_indices - integer_indices) > 0.0).mean()

            details['quality'].update(
                index_deviation=index_deviation,  # Deviation from integer values
                max_deviation=max_deviation,  # Maximum acceptable deviation in cluster indices
                half_percent=half_index_pct,  # Percent of indices closer to 0.5
            )

            subtree_ratio = 0.0
            if len(details['subtrees']) > 1:
                subtree_ratio = round(details['subtrees'][1] / details['subtrees'][0], 1)
            details['quality'].update(subtree_ratio=subtree_ratio)

            # Diagnoses
            axis_is_inverted = (
                misfit_percent > 75 and
                index_deviation >= 2 * max_deviation and
                details['quality']['pixel_error'] < 3.0
            )
            solution_is_poor = (
                misfit_percent > 25 and
                details['quality']['pixel_error'] > 1.5 * max(details['quality']['expected_pixel_error'], 3.0)
            )
            wrong_spot_parameters = (
                details['quality']['half_percent'] > 25 and
                details['quality']['subtree_ratio'] < .1
            )
            fractional_indexing = (
                details['quality']['half_percent'] > 25 and
                details['quality']['subtree_ratio'] > 0.85
            )

            if fractional_indexing:
                problems |= IndexProblem.FRACTIONAL_INDICES
            elif index_deviation > max_deviation:
                problems |= IndexProblem.NON_INTEGRAL_INDICES
            elif details['quality']['subtree_ratio'] > MAX_SUBTREE_RATIO:
                problems |= IndexProblem.MULTIPLE_SUBTREES
            elif wrong_spot_parameters:
                problems |= IndexProblem.WRONG_SPOT_PARAMETERS

            if axis_is_inverted:
                problems |= IndexProblem.INVERTED_AXIS
            elif solution_is_poor:
                problems |= IndexProblem.POOR_SOLUTION

            for message in ['message_1', 'message_2', 'message_3']:
                failure = get_failure(log_details.get(message), INDEX_FAILURES)
                problems |= failure

        state_messages = [INDEX_STATES[problem] for problem in problems.values() if problem in INDEX_STATES]
        if state_messages:
            states, messages = zip(*state_messages)
            state = max(states)
        else:
            state, messages = StateType.SUCCESS, ()
        return Result(state=state, messages=messages, flags=problems, details=details)

    @classmethod
    def parse_xscale(cls):
        """
        Harvest data from XSCALE
        """
        if not os.path.exists('XSCALE.LP'):
            return {}

        with open('XSCALE.LP', 'r') as handle:
            data = handle.read()

        # extract separate sections corresponding to different datasets
        header = "\n".join(re.findall(r'(CONTROL CARDS.+?CORRECTION FACTORS AS FUNCTION)', data, re.DOTALL))
        section = re.compile(
            r'(STATISTICS OF SCALED OUTPUT DATA SET :\s+(\S+).+? ========== )STATISTICS OF INPUT DATA SET', re.DOTALL
        )
        footer = re.compile(
            r'(WILSON STATISTICS OF SCALED DATA SET:\s+(\S+).+?HIGHER ORDER MOMENTS OF WILSON)', re.DOTALL
        )
        footers = {key: body for body, key in footer.findall(data)}

        data_sections = {
            key: "\n".join([header, body, footers[key]])
            for body, key in section.findall(data)
        }

        lexicon = cls.get_lexicon('XSCALE.LP')
        return {
            key: cls.parse_text(info, lexicon)
            for key, info in data_sections.items()
        }

