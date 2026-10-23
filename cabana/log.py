"""Lightweight logging facade.

A thin wrapper around stdlib :mod:`logging`. The public surface
(``Log.logger``, ``Log.init_log_path(path)``, ``Log.log_parameters(param_path)``)
is kept so call sites do not need to change.

``Log.logger`` is the ``cabana`` logger (module loggers such as
``logging.getLogger(__name__)`` are its children), so the root logger and other
libraries' logging are left alone. The console shows INFO and above in colour;
the log file records DEBUG and above in plain text with timestamps.
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
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(_CONSOLE_FORMATTER)
    logger.addHandler(sh)
    return logger


class _LogFacade:
    """Module-level singleton providing the historical ``Log`` API."""

    def __init__(self):
        self.log_path = None
        self.log_name = None
        self._file_handler = None
        self.logger = _build_logger()

    def init_log_path(self, log_path=None):
        """Start a new timestamped log file in ``log_path``.

        Any log file opened by a previous call is closed first, so each run
        writes only to its own file.
        """
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(self.log_path, exist_ok=True)

        self.log_name = os.path.join(
            self.log_path, '%s.log' % time.strftime('%Y-%m-%d-%H-%M-%S'))

        self.close_log_file()
        fh = logging.handlers.RotatingFileHandler(
            filename=self.log_name, mode='a',
            maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FILE_FORMATTER)
        self.logger.addHandler(fh)
        self._file_handler = fh

    def close_log_file(self):
        """Detach and close the current log file, if any."""
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
            data = yaml.safe_load(pf)
            for line in yaml.dump(data, default_flow_style=False).split("\n"):
                self.logger.debug(line)
        footer = '*' * ((len(header) - 3) // 2) + "End" + '*' * ((len(header) - 3) // 2)
        self.logger.debug(footer)


Log = _LogFacade()
