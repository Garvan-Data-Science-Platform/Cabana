"""Lightweight logging facade.

A thin wrapper around stdlib :mod:`logging`. The public surface
(``Log.logger``, ``Log.init_log_path(path)``, ``Log.log_parameters(param_path)``)
is kept so call sites do not need to change.

``Log.logger`` is the ``cabana`` logger (module loggers such as
``logging.getLogger(__name__)`` are its children), so the root logger and other
libraries' logging are left alone. The console shows INFO and above in colour;
log files record DEBUG and above in plain text with timestamps.

Two kinds of log file:

* ``Log.init_log_path(dir)`` opens the session log for a script or the tests;
  calling it again replaces the previous session log.
* ``Log.run_log(dir, param_path)`` opens a per-run log (one file per batch or
  single-image run, in the run's output folder). It is independent of the
  session log: both receive every message while the run log is open, and
  closing the run log leaves the session log in place.
"""

import os
import time
import logging
import logging.handlers

import yaml
import colorlog

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

LOGGER_NAME = 'cabana'

_LOG_COLORS = {
    'DEBUG': 'green',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}
_CONSOLE_FORMATTER = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]- %(message)s', log_colors=_LOG_COLORS)
_FILE_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def _build_logger():
    """Return the ``cabana`` logger with a console handler attached.

    Always usable from import time so callers can log without first calling
    :func:`init_log_path`.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)
    # The console handler lives here, so do not hand records to the root
    # logger as well: a root handler (logging.basicConfig in a script or
    # notebook) would print every line twice.
    logger.propagate = False
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(_CONSOLE_FORMATTER)
    logger.addHandler(sh)
    return logger


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


class RunLog:
    """The log file of one run; see :meth:`_LogFacade.run_log`.

    Close it with :meth:`close` or by using it as a context manager. An
    exception leaving the ``with`` block is recorded with its traceback
    before the file is closed.
    """

    def __init__(self, logger, log_dir, param_path=None):
        self._logger = logger
        self._handler = _new_log_file(log_dir)
        self.path = self._handler.baseFilename
        logger.addHandler(self._handler)
        try:
            logger.info('Log file: {}'.format(self.path))
            if param_path:
                Log.log_parameters(param_path)
        except Exception:
            self.close()
            raise

    def close(self):
        """Detach and close the file; safe to call more than once."""
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self._logger.exception('Run failed')
        self.close()
        return False


class _LogFacade:
    """Module-level singleton providing the historical ``Log`` API."""

    def __init__(self):
        self.log_path = None
        self.log_name = None
        self._file_handler = None
        self.logger = _build_logger()

    def init_log_path(self, log_path=None):
        """Start a new timestamped session log file in ``log_path``.

        A session log opened by a previous call is closed first. Run logs
        (:meth:`run_log`) are not affected.
        """
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'logs')
        self.close_log_file()
        fh = _new_log_file(self.log_path)
        self.log_name = fh.baseFilename
        self.logger.addHandler(fh)
        self._file_handler = fh

    def run_log(self, log_dir, param_path=None):
        """Open a :class:`RunLog` in ``log_dir``, recording ``param_path`` in it."""
        return RunLog(self.logger, log_dir, param_path)

    def close_log_file(self):
        """Detach and close the session log file, if any."""
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def log_parameters(self, param_path):
        """Write the parameter file to the log file (DEBUG, not shown on the console)."""
        if not os.path.exists(param_path):
            self.logger.error("{} not exists.".format(param_path))
            return
        header = "********{}********".format(os.path.basename(param_path))
        self.logger.debug(header)
        with open(param_path, 'r') as pf:
            try:
                data = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                self.logger.error("Could not parse {}: {}".format(param_path, exc))
                return
            for line in yaml.dump(data, default_flow_style=False).split("\n"):
                self.logger.debug(line)
        footer = '*' * ((len(header) - 3) // 2) + "End" + '*' * ((len(header) - 3) // 2)
        self.logger.debug(footer)


Log = _LogFacade()
