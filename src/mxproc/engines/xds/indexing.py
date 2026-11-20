from __future__ import annotations

import re
from enum import auto
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy
from numpy.typing import NDArray
from parsefire import parser

from mxproc.command import CommandFailed, CommandRunner, LocalRunner
from mxproc.common import (
    Flag,
    Result,
    StateType,
    backup_files,
    generate_failure,
    show_warnings,
)
from mxproc.log import logger
from mxproc.xtal import Experiment, Lattice, POINT_GROUPS, lattice_point_groups

from . import io
from .parser import XDSParser

MAX_INDEX_TRIES = 3
DEFAULT_MIN_SIGMA = 4
MAX_SUBTREE_RATIO = 0.05


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
    IndexProblem.WRONG_SPOT_PARAMETERS: (StateType.WARNING, 'Spot are closer than allowed'),
}


def get_failure(message: str | None, failures: dict[str, IndexProblem]) -> IndexProblem:
    """
    Search for matching failures in a dictionary of pattern, value pairs and return the corresponding
    failure code or IndexProblem.NONE if not found.
    """
    if message:
        for pattern, problem in failures.items():
            if re.match(pattern, message):
                return problem
    return IndexProblem.NONE


def max_sub_range(values: NDArray, size: int) -> Tuple[int, int]:
    """
    Find the range of values of a given size which maximizes the sum of the sub array.
    """
    max_index = int(numpy.argmax([values[i:i + size].sum() for i in range(len(values) - size)]))
    return max_index, max_index + size


def get_spot_distribution(work_dir: Path = Path(".")) -> Any:
    """
    Calculate the percentage of indexed spots per image.
    """
    data = numpy.loadtxt(Path(work_dir) / 'SPOT.XDS', comments='!')
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
    results[:, 0] = ((edges[1:] + edges[:-1]) * 0.5).astype(int)
    results[:, 1] = total
    results[:, 2] = indexed_counts

    fraction_indexed = numpy.divide(
        indexed_counts, total, out=numpy.zeros(total.shape, dtype=numpy.float64), where=total != 0
    )
    range_start, range_end = max_sub_range(fraction_indexed, len(fraction_indexed) // 4)

    return results, (results[range_start, 0], results[range_end, 0])


class IndexParamManager:
    """
    Stores and manages indexing parameters across multiple trials, updating them
    based on problems detected during the previous iteration.
    """

    degrees: dict
    params: dict
    min_sigma: float
    max_sigma: float

    def __init__(
        self,
        data_range: Sequence[Tuple[int, int]],
        spot_range: Sequence[Tuple[int, int]],
        min_sigma: float = DEFAULT_MIN_SIGMA,
        max_sigma: float = 100,
        **extras,
    ):
        self.degrees = {}
        self.params = {'spot_range': spot_range, "data_range": data_range, **extras}
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def get_degree(self, name: str) -> int:
        self.degrees[name] = self.degrees.get(name, 0) + 1
        return self.degrees[name]

    def get_parameters(self) -> dict:
        return self.params

    def update_parameters(
        self,
        flags: IndexProblem,
        spot_range: Sequence[Tuple[int, int]] | None = None,
    ) -> Tuple[bool, Sequence[str]]:
        messages = []
        request_retry = False

        if IndexProblem.INVERTED_AXIS in flags:
            self.params['invert_spindle'] = True
            messages.append('Inverting the rotation direction')
            if self.get_degree('invert_spindle') == 1:
                request_retry = True

        if IndexProblem.WRONG_SPOT_PARAMETERS in flags:
            self.params['spot_size'] = 0.5 / self.get_degree('spot_size')
            messages.append("Reducing accepted spot size & separation")
            request_retry = True

        if (IndexProblem.LOW_INDEXED_FRACTION | IndexProblem.POOR_SOLUTION) in flags and spot_range:
            self.params['spot_range'] = [spot_range]
            messages.append("Changing spot range")
            request_retry = True
        elif IndexProblem.POOR_SOLUTION in flags:
            self.params['error_scale'] = 2.0 * self.get_degree('error_scale')
            self.params['spot_separation'] = 1.0 - 0.25 ** self.get_degree('spot_separation')
            self.params['spot_size'] = 1.0 + 0.5 * self.get_degree('spot_size')
            messages.append("Relaxing quality criteria")
            request_retry = True

        if IndexProblem.REFINEMENT_FAILED in flags:
            refinement_flags = ('CELL', 'BEAM', 'ORIENTATION', 'AXIS',)[:-self.get_degree('refine_index')]
            if len(refinement_flags) > 1:
                self.params['refine_index'] = refinement_flags
                messages.append("Adjusting refinement parameters")
                request_retry = True

        if IndexProblem.INSUFFICIENT_SPOTS in flags:
            self.min_sigma = DEFAULT_MIN_SIGMA - self.get_degree('min_sigma')
            request_retry = True

        return request_retry, messages


def diagnose_indexing(
    log_details: dict,
    param_details: dict,
    spot_counts: Any = None,
    best_range: Any = None,
) -> Result:
    """
    Evaluate parsed IDXREF.LP and XPARM.XDS data into a structured Result with IndexProblem flags.
    """
    details: dict[str, Any] = {}
    problems = IndexProblem.NONE
    params = dict(param_details)

    lattice = Lattice(params.pop('sg_number', 1), *params.pop('unit_cell', ()))
    details.update(params)
    details['lattice'] = lattice
    details['subtrees'] = [tree['population'] for tree in log_details.get('subtrees', [])]
    details['overlaps'] = log_details.get('delta_overlaps', [])
    details['index_origins'] = log_details.get('index_origins', [])
    details['lattices'] = log_details.get('lattices', [])
    details['point_groups'] = lattice_point_groups(*[lat['character'] for lat in details['lattices']])

    best_lattice = details['lattices'][-1]
    best_spacegroup = POINT_GROUPS[best_lattice['character']][-1]

    details['symmetry_lattice'] = Lattice(best_spacegroup, *best_lattice['unit_cell'])
    details['quality'] = dict(log_details.get('quality', {}))

    exp_ang_err = log_details.get('exp_ang_err', 2.0)
    exp_pos_err = log_details.get('exp_pos_err', 3.0)

    details['spots'] = dict(log_details.get('spots', {}))
    details['spots'].update(counts=spot_counts, best_range=best_range)

    misfit_percent = 100 * details['spots'].get('misfits', 0) / details['spots'].get('total', 1)
    details['quality'].update(
        misfit_percent=misfit_percent,
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
            index_deviation = float(numpy.abs(indices - integer_indices).mean())
            half_index_pct = float(100 * (numpy.abs(fractional_indices - integer_indices) > 0.0).mean())

    details['quality'].update(
        index_deviation=index_deviation,
        max_deviation=max_deviation,
        half_percent=half_index_pct,
    )

    subtree_ratio = 0.0
    if len(details['subtrees']) > 1:
        subtree_ratio = round(details['subtrees'][1] / details['subtrees'][0], 1)
    details['quality'].update(subtree_ratio=subtree_ratio)

    axis_is_inverted = (
        misfit_percent > 75
        and index_deviation >= 2 * max_deviation
        and details['quality']['pixel_error'] < 3.0
    )
    solution_is_poor = (
        misfit_percent > 25
        and details['quality']['pixel_error'] > 1.5 * max(details['quality']['expected_pixel_error'], 3.0)
    )
    wrong_spot_parameters = (
        details['quality']['half_percent'] > 25
        and details['quality']['subtree_ratio'] < 0.1
    )
    fractional_indexing = (
        details['quality']['half_percent'] > 25
        and details['quality']['subtree_ratio'] > 0.85
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
        problems |= get_failure(log_details.get(message), INDEX_FAILURES)

    return _build_index_result(problems, details)


def _build_index_result(problems: IndexProblem, details: dict) -> Result:
    state_messages = [INDEX_STATES[problem] for problem in problems.values() if problem in INDEX_STATES]
    if state_messages:
        states, messages = zip(*state_messages)
        state = max(states)
    else:
        state, messages = StateType.SUCCESS, ()
    return Result(state=state, messages=messages, flags=problems, details=details)


def auto_index(
    experiment: Experiment,
    options: dict,
    work_dir: Path = Path("."),
    runner: CommandRunner | None = None,
    max_tries: int = MAX_INDEX_TRIES,
) -> Result:
    """
    Run XDS auto-indexing for an experiment with iterative diagnosis and parameter remediation.
    """
    work_dir = Path(work_dir)
    runner = runner if runner is not None else LocalRunner()
    manager = IndexParamManager(**options)
    result = generate_failure('')

    for trial in range(max_tries):
        backup_files(str(work_dir / 'IDXREF.LP'))
        backup_files(str(work_dir / 'SPOT.XDS'))
        parameters = io.XDSParameters(**manager.get_parameters())
        io.filter_spots(min_sigma=manager.min_sigma, max_sigma=manager.max_sigma, work_dir=work_dir)
        io.create_input_file(('IDXREF',), experiment, parameters, work_dir=work_dir)

        try:
            runner.run('xds_par', work_dir, desc=f'- Attempt #{trial + 1} of auto-indexing')
            log_details = XDSParser.parse(str(work_dir / 'IDXREF.LP'))
            param_details = XDSParser.parse(str(work_dir / 'XPARM.XDS'), silent=True)
            spot_counts, best_range = get_spot_distribution(work_dir=work_dir)
            result = diagnose_indexing(log_details, param_details, spot_counts=spot_counts, best_range=best_range)
        except (parser.FilesMissing, parser.MissingLexicon) as err:
            logger.exception(err)
            result = _build_index_result(IndexProblem.SOFTWARE_ERROR, {})
        except (CommandFailed, FileNotFoundError, KeyError) as err:
            result = generate_failure(f"Command failed: {err}")

        request_retry = (
            result.state == StateType.FAILURE
            and IndexProblem.SOFTWARE_ERROR not in IndexProblem(result.flags)
        )
        if not request_retry:
            break

        show_warnings('Indexing Problems:', result.messages)
        kw = {}
        if 'spots' in result.details:
            kw['spot_range'] = result.details['spots'].get('best_range')
        request_retry, retry_messages = manager.update_parameters(IndexProblem(result.flags), **kw)
        if not request_retry:
            break
        show_warnings('Retrying Indexing:', retry_messages)

    return result
