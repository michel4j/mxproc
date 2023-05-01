import asyncio
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Sequence

from tqdm import tqdm

from mxproc.log import logger


class CommandNotFound(Exception):
    ...


class CommandFailed(Exception):
    ...


def files_exist(files: Sequence[str]) -> bool:
    """
    Check if a list of files exist on disk
    :param files: files to check
    :return: True if they exist, false otherwise
    """

    for file in files:
        if not Path(file).exists():
            return False

    return True


class Command:
    def __init__(self, shell_cmd: str, logfile: Union[str, Path] = "commands.log", desc: str = "", check_files: Sequence[str] = ()):
        """
        Objects for running commands

        :param shell_cmd: command arguments
        :param logfile: destination of standard output including errors
        :param desc: descriptive label of command
        :param check_files: At the end of the command these files must exist, otherwise the command failed
        """
        self.outfile = Path(logfile)
        self.shell_cmd = shell_cmd
        self.label = desc
        self.check_files = check_files

    async def exec(self):
        """
        Main method to run the command asynchronously and update the progress bar with a descriptive label
        """

        with open(self.outfile, 'a') as stdout:
            start_time = time.time()
            start_str = datetime.now().strftime('%H:%M:%S')
            bar_fmt = "{desc: <83}{elapsed}{postfix}"
            with tqdm(desc=f"{start_str} {self.label} ... ", miniters=1, leave=False, bar_format=bar_fmt) as spinner:
                proc = await asyncio.create_subprocess_shell(self.shell_cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stdout)
                while proc.returncode is None:
                    spinner.update()
                    await asyncio.sleep(.05)
            elapsed = time.time() - start_time

            if proc.returncode != 0 or not files_exist(self.check_files):
                logger.error_value(self.label, "[FAILED]", spacer=' ')
                raise CommandFailed(self.shell_cmd)
            else:
                logger.info(self.label)

    def run_sync(self):
        """
        Run command synchronously
        :return:
        """
        try:
            logger.info_value(f"{self.label}", '.')
            with open(self.outfile, 'a') as stdout:
                output = subprocess.check_output(self.shell_cmd, shell=True)
                stdout.write(output)

        except (subprocess.CalledProcessError, CommandFailed) as err:
            raise CommandFailed(f"{err}")

    def run_async(self):
        """
        Run command in an event loop
        :return:
        """
        loop = asyncio.get_event_loop()
        try:
            tasks = [loop.create_task(self.exec())]
            wait_tasks = asyncio.gather(*tasks)
            loop.run_until_complete(wait_tasks)
        except (subprocess.CalledProcessError, CommandFailed) as err:
            raise CommandFailed(f"{err}")


def run_command(cmd, desc: str = "", logfile: Union[str, Path] = "commands.log", sync=False, check_files: Sequence[str] = ()):
    """
    Creates and executes a command instance

    :param cmd: command arguments
    :param logfile: destination of standard output including errors
    :param desc: descriptive label of command
    :param sync: Run synchronously, default False
    :param check_files: At the end of the command these files must exist, otherwise the command failed
    """

    command = Command(cmd, desc=desc, logfile=logfile, check_files=check_files)
    if sync:
        command.run_sync()
    else:
        command.run_async()
