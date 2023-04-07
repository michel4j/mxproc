from typing import Tuple, Sequence
from mxproc.common import Result, StateType, generate_failure, show_warnings
from mxproc.command import run_command, CommandFailed
from mxproc.xtal import Experiment

from .parser import XDSParser, IndexProblem
from . import io


def autoindex_trial(experiment: Experiment, manager, trial: int) -> Tuple[Result, bool, Sequence[str]]:
    """
    Run one trials of auto indexing
    :param experiment: target Experiment for the trial
    :param manager: Parameter manager
    :param trial: integer representing the trial number
    :return:
    """

    parameters = io.XDSParameters(**manager.get_parameters())
    io.filter_spots(min_sigma=manager.min_sigma, max_sigma=manager.max_sigma)
    io.create_input_file(('IDXREF',), experiment, parameters)
    try:
        run_command('xds_par', desc=f'- Attempt #{trial + 1} of auto-indexing')
        result = XDSParser.parse_index()
    except (CommandFailed, FileNotFoundError, KeyError) as err:
        result = generate_failure(f"Command failed: {err}")

    retry_messages = []

    request_retry = (
            result.state == StateType.FAILURE and
            IndexProblem.SOFTWARE_ERROR not in result.flags
    )

    if request_retry:
        show_warnings(f'Indexing Problems:', result.messages)
        request_retry, retry_messages = manager.update_parameters(
            result.flags, spot_range=result.details['spots']['best_range']
        )

    return result, request_retry, retry_messages
