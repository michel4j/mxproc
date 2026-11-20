from pathlib import Path
from typing import Tuple, Sequence
from mxproc.common import Result, StateType, generate_failure, show_warnings, backup_files
from mxproc.command import CommandFailed, CommandRunner, LocalRunner
from mxproc.xtal import Experiment

from .parser import XDSParser, IndexProblem
from . import io


def autoindex_trial(
    experiment: Experiment,
    manager,
    trial: int,
    work_dir: Path = Path("."),
    runner: CommandRunner | None = None,
) -> Tuple[Result, bool, Sequence[str]]:
    """
    Run one trials of auto indexing
    :param experiment: target Experiment for the trial
    :param manager: Parameter manager
    :param trial: integer representing the trial number
    :param work_dir: working directory for indexing files
    :param runner: optional CommandRunner for running xds_par
    :return:
    """
    work_dir = Path(work_dir)
    runner = runner if runner is not None else LocalRunner()

    parameters = io.XDSParameters(**manager.get_parameters())
    backup_files(str(work_dir / 'SPOT.XDS'))
    io.filter_spots(min_sigma=manager.min_sigma, max_sigma=manager.max_sigma, work_dir=work_dir)
    io.create_input_file(('IDXREF',), experiment, parameters, work_dir=work_dir)
    try:
        runner.run('xds_par', work_dir, desc=f'- Attempt #{trial + 1} of auto-indexing')
        result = XDSParser.parse_index(work_dir=work_dir)
    except (CommandFailed, FileNotFoundError, KeyError) as err:
        result = generate_failure(f"Command failed: {err}")

    retry_messages = []
    request_retry = (
            result.state == StateType.FAILURE
            and IndexProblem.SOFTWARE_ERROR not in IndexProblem(result.flags)
    )

    if request_retry:
        show_warnings(f'Indexing Problems:', result.messages)
        kw = {}
        if 'spots' in result.details:
            kw['spot_range'] = result.details['spots']['best_range']

        request_retry, retry_messages = manager.update_parameters(
            IndexProblem(result.flags), **kw
        )

    return result, request_retry, retry_messages
