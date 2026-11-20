
import textwrap
import logging
from logging.handlers import RotatingFileHandler


IMPORTANT = 25
logging.addLevelName(IMPORTANT, 'IMPORTANT')


class TermColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALICS = '\033[3m'

    @classmethod
    def bold(cls, text):
        return f'{cls.BOLD}{text}{cls.END}'

    @classmethod
    def warn(cls, text):
        return f'{cls.WARNING}{text}{cls.END}'

    @classmethod
    def success(cls, text):
        return f'{cls.GREEN}{text}{cls.END}'

    @classmethod
    def error(cls, text):
        return f'{cls.FAIL}{text}{cls.END}'

    @classmethod
    def emphasis(cls, text):
        return f'{cls.BOLD}{text}{cls.END}'

    @classmethod
    def debug(cls, text):
        return f'{cls.BLUE}{text}{cls.END}'

    @classmethod
    def normal(cls, text):
        return text

    @classmethod
    def underline(cls, text):
        return f'{cls.UNDERLINE}{text}{cls.END}'

    @classmethod
    def italics(cls, text):
        return f'{cls.ITALICS}{text}{cls.END}'


class NullHandler(logging.Handler):
    """A do-nothing log handler."""

    def emit(self, record):
        pass


class ColoredConsoleHandler(logging.StreamHandler):
    """
    A colored console log handler
    """

    def format(self, record):
        msg = super(ColoredConsoleHandler, self).format(record)
        if record.levelno == logging.WARNING:
            msg = TermColor.warn(msg)
        elif record.levelno > logging.WARNING:
            msg = TermColor.error(msg)
        elif record.levelno == logging.DEBUG:
            msg = TermColor.debug(msg)
        elif record.levelno == IMPORTANT:
            msg = TermColor.emphasis(msg)
        return msg


def get_module_logger(name: str = __name__) -> logging.Logger:
    """
    Create a logger for the given name and return it
    :param name: string name for the logger
    :return: logging.Logger instance
    """
    name = name.split('.')[-1]
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(NullHandler())
    return log


def log_to_console(level: int = logging.INFO):
    """
    Add a console log handler

    :param level: log level for the file
    """
    console = ColoredConsoleHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_to_file(filename: str, level: int = logging.DEBUG):
    """
    Add a rotating file handler

    :param filename: name of log file
    :param level: log level for the file
    """
    logfile = RotatingFileHandler(filename, maxBytes=1000000, backupCount=10)
    logfile.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%b/%d %H:%M:%S')
    logfile.setFormatter(formatter)
    logging.getLogger('').addHandler(logfile)


def log_value(descr, value, style=TermColor.bold, width=79, spacer='.'):
    """
    Format a log line of the form ' Description: ............... value '

    :param descr: description text
    :param value: value text
    :param style: further style function to apply default is TermColor.bold
    :param width: total width of line
    :param spacer: spacer character, default '.'
    :returns: formatted text
    """
    value_width = width - len(descr) - 1
    value_texts = textwrap.wrap(value, value_width)
    if value_texts:
        return [
            f'{descr} {spacer * (width - len(descr) - len(line) - 2)} {style(line)}'
            for line in value_texts
        ]
    else:
        return [f'{descr} {spacer * (width - len(descr) - 1)}']


class LoggerManager:
    logger: logging.Logger

    def __init__(self):
        self.logger = get_module_logger("mxproc")

    def __getattr__(self, item):
        if hasattr(self.logger, item):
            return getattr(self.logger, item)
        raise AttributeError

    def line(self, kind: str = '-'):
        """
        Draw a horizontal line in the log
        :param kind: line character
        """
        self.logger.info(kind * 79)

    def banner(self, text: str, overline: bool = True, underline: bool = True, line: str = '='):
        """
        Display text in a banner with optional lines
        :param text:  Banner text, will be centered in the 80 character window
        :param overline: Whether to draw lines above
        :param underline: Whether to draw lines below
        :param line: Line Character
        """
        if overline:
            self.line(line)
        self.logger.info(f'{text:^79}')
        if underline:
            self.line(line)

    def info_value(self, text, value_text, spacer='.'):
        """
        Display a value at the end of the info log line, wrap as necessary.
        :param text:  log message body
        :param value_text: log message value
        :param spacer: line character
        """
        for line in log_value(text, value_text, spacer=spacer):
            self.info(line)

    def error_value(self, text, value_text, spacer='.'):
        """
        Display a value at the end of the error log line, wrap as necessary.
        :param text:  log message body
        :param value_text: log message value
        :param spacer: line character
        """
        for line in log_value(text, value_text, spacer=spacer):
            self.error(line)

    def warning_value(self, text, value_text, spacer='.'):
        """
        Display a value at the end of the warning log line, wrap as necessary.
        :param text:  log message body
        :param value_text: log message value
        :param spacer: line character
        """
        for line in log_value(text, value_text, spacer=spacer):
            self.warning(line)


logger = LoggerManager()

