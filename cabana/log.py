"""Logging for CABANA.

Library code logs to the ``cabana`` logger (``Log.logger``) or to module
loggers (``logging.getLogger(__name__)``, children of ``cabana``) and attaches
no output when imported, the standard practice for Python libraries. Where
messages go is decided by the application:

* :func:`setup_logging` is called once by the GUI and the command lines (and by
  scripts that want console output): a coloured console at INFO and, if a
  folder is given, a timestamped DEBUG log file there.
* :func:`run_log` is used by ``Cabana.run`` and ``BatchProcessor.run``: for the
  duration of a ``with`` block, the calling thread's messages are also written
  to a new file in the run's ``Logs`` folder.

The ``cabana`` logger is set to INFO at import so a run log records progress
even when the host has not set up logging; :func:`setup_logging` lowers it to
DEBUG so log files get the stage details too.
"""

import os
import time
import logging
import threading
import contextlib
import logging.handlers

import yaml
import colorlog

LOGGER_NAME = 'cabana'

logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

_CONSOLE_FORMATTER = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]- %(message)s',
    log_colors={'DEBUG': 'green', 'INFO': 'white', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'red'})
_FILE_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Handlers added by setup_logging, replaced when it is called again.
_app_handlers = []


def _new_log_file(log_dir):
    """Return a DEBUG file handler on a new, uniquely named file in ``log_dir``."""
    os.makedirs(log_dir, exist_ok=True)
    stem = os.path.join(log_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
    path, n = stem + '.log', 1
    while os.path.exists(path):           # another run started this second
        path = '%s-%d.log' % (stem, n)
        n += 1
    fh = logging.handlers.RotatingFileHandler(
        filename=path, mode='a',
        maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FORMATTER)
    return fh


def setup_logging(log_dir=None):
    """Send log messages to the console and, with ``log_dir``, to a log file.

    For applications and scripts; library code never calls it. Handlers go on
    the root logger, so warnings from other libraries are shown too. Calling it
    again replaces the handlers of the previous call. Returns the log file's
    path, or None.
    """
    root = logging.getLogger()
    for h in _app_handlers:
        root.removeHandler(h)
        h.close()
    _app_handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(_CONSOLE_FORMATTER)
    _app_handlers.append(console)
    path = None
    if log_dir:
        fh = _new_log_file(log_dir)
        path = fh.baseFilename
        _app_handlers.append(fh)
    for h in _app_handlers:
        root.addHandler(h)

    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    return path


@contextlib.contextmanager
def run_log(log_dir, param_path=None):
    """Write this thread's ``cabana`` messages to a new file in ``log_dir``.

    Records ``param_path`` at the start and any exception (including Ctrl+C)
    leaving the block. Yields the file's path.
    """
    handler = _new_log_file(log_dir)
    thread_id = threading.get_ident()
    # Only this run: other GUI actions log from other threads meanwhile
    handler.addFilter(lambda record: record.thread == thread_id)
    logger.addHandler(handler)
    try:
        logger.info('Log file: %s', handler.baseFilename)
        if param_path:
            log_parameters(param_path)
        yield handler.baseFilename
    except (Exception, KeyboardInterrupt):
        logger.exception('Run failed')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def log_parameters(param_path):
    """Write the parameter file to the log (DEBUG: log files only)."""
    if not os.path.exists(param_path):
        logger.error("{} not exists.".format(param_path))
        return
    header = "********{}********".format(os.path.basename(param_path))
    logger.debug(header)
    with open(param_path, 'r') as pf:
        try:
            data = yaml.safe_load(pf)
        except yaml.YAMLError as exc:
            logger.error("Could not parse {}: {}".format(param_path, exc))
            return
    for line in yaml.dump(data, default_flow_style=False).split("\n"):
        logger.debug(line)
    footer = '*' * ((len(header) - 3) // 2) + "End" + '*' * ((len(header) - 3) // 2)
    logger.debug(footer)


class _LogFacade:
    """The historical ``Log`` API, kept so call sites need not change."""

    logger = logger
    log_name = None          # session log file of the last init_log_path call

    def init_log_path(self, log_path=None):
        """Same as ``setup_logging(log_path)``; defaults to ``cabana/logs``."""
        if not log_path:
            log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        self.log_name = setup_logging(log_path)

    run_log = staticmethod(run_log)
    log_parameters = staticmethod(log_parameters)


Log = _LogFacade()
