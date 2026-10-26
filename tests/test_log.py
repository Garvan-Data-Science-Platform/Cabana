"""Tests for ``cabana.log``: library logger, setup_logging and run logs."""

import os
import glob
import logging
import threading

import yaml
import pytest

from cabana import log as cabana_log
from cabana.log import Log, LOGGER_NAME, setup_logging, run_log


@pytest.fixture
def app_logging():
    """Undo setup_logging's root handlers and level after the test."""
    level = cabana_log.logger.level
    yield
    root = logging.getLogger()
    for h in cabana_log._app_handlers:
        root.removeHandler(h)
        h.close()
    cabana_log._app_handlers.clear()
    cabana_log.logger.setLevel(level)


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _run_handlers():
    return [h for h in Log.logger.handlers if isinstance(h, logging.FileHandler)]


def test_library_logger_attaches_no_output():
    """Importing cabana must not print or write anything by itself."""
    assert Log.logger.name == LOGGER_NAME
    assert all(isinstance(h, logging.NullHandler) for h in Log.logger.handlers)
    assert Log.logger.propagate


def test_records_reach_root_handlers_once(app_logging):
    """Host handlers (caplog, a script's basicConfig) see each record once."""
    seen = []
    host = logging.Handler()
    host.emit = seen.append
    logging.getLogger().addHandler(host)
    try:
        Log.logger.info('hello')
    finally:
        logging.getLogger().removeHandler(host)
    assert [r.getMessage() for r in seen] == ['hello']


def test_setup_logging_writes_plain_timestamped_file(app_logging, tmp_path):
    path = setup_logging(str(tmp_path))
    Log.logger.warning('careful')
    Log.logger.debug('details')
    logging.getLogger('cabana.scores').info('from scores')
    logging.getLogger('PIL').info('third-party info')       # PIL stays at WARNING: dropped
    text = _read(path)
    assert '\x1b[' not in text
    assert '[WARNING] cabana: careful' in text
    assert '[DEBUG] cabana: details' in text
    assert '[INFO] cabana.scores: from scores' in text
    assert 'third-party info' not in text


def test_setup_logging_again_replaces_its_handlers(app_logging, tmp_path):
    first = setup_logging(str(tmp_path / 'a'))
    second = setup_logging(str(tmp_path / 'b'))
    Log.logger.info('after second call')
    assert 'after second call' not in _read(first)
    assert 'after second call' in _read(second)
    assert len(cabana_log._app_handlers) == 2          # one console, one file


def test_run_log_records_parameters_and_closes(tmp_path):
    params = tmp_path / 'p.yml'
    yaml.safe_dump({'Detection': {'Min Line Width': 5}}, open(params, 'w'))
    with run_log(str(tmp_path / 'Logs'), str(params)) as path:
        Log.logger.info('during run')
    Log.logger.info('after run')
    text = _read(path)
    assert 'during run' in text and 'Min Line Width' not in text   # INFO by default
    assert 'after run' not in text
    assert _run_handlers() == []


def test_run_log_gets_debug_details_after_setup_logging(app_logging, tmp_path):
    params = tmp_path / 'p.yml'
    yaml.safe_dump({'Detection': {'Min Line Width': 5}}, open(params, 'w'))
    setup_logging()
    with run_log(str(tmp_path / 'Logs'), str(params)) as path:
        pass
    assert 'Min Line Width: 5' in _read(path)


def test_run_log_records_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with run_log(str(tmp_path)) as path:
            raise RuntimeError('boom')
    text = _read(path)
    assert 'Run failed' in text and 'RuntimeError: boom' in text
    assert _run_handlers() == []


def test_run_log_ignores_other_threads(tmp_path):
    """GUI actions on other threads must not land in a running batch's log."""
    with run_log(str(tmp_path)) as path:
        t = threading.Thread(target=lambda: Log.logger.warning('from another thread'))
        t.start()
        t.join()
        Log.logger.info('from the run')
    text = _read(path)
    assert 'from the run' in text and 'from another thread' not in text


def test_run_logs_in_the_same_second_get_distinct_files(tmp_path):
    with run_log(str(tmp_path)) as a, run_log(str(tmp_path)) as b:
        assert a != b
    assert len(glob.glob(str(tmp_path / '*.log'))) == 2


def test_bad_parameter_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / 'bad.yml'
    bad.write_text('Configs: [unclosed')
    with run_log(str(tmp_path / 'Logs'), str(bad)) as path:
        pass
    assert 'Could not parse' in _read(path)


class _FakeBatchCabana:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        pass


def _batch_inputs(tmp_path):
    import numpy as np
    import cv2
    input_dir = tmp_path / 'in'
    input_dir.mkdir()
    cv2.imwrite(str(input_dir / 'a.png'), np.full((40, 40, 3), 200, np.uint8))
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return input_dir, out_dir, os.path.join(root, 'cabana', 'default_params.yml')


@pytest.mark.parametrize('logs_name', ['Logs', 'logs'])
def test_batch_log_lives_in_output_and_survives_a_new_run(tmp_path, monkeypatch, logs_name):
    import cabana.batch as batch_mod
    from cabana.batch import BatchProcessor

    input_dir, out_dir, param_file = _batch_inputs(tmp_path)
    (out_dir / 'stale.txt').write_text('old result')
    (out_dir / 'StaleDir').mkdir()
    (out_dir / logs_name).mkdir()
    (out_dir / logs_name / 'old.log').write_text('earlier run')
    monkeypatch.setattr(batch_mod, 'BatchCabana', _FakeBatchCabana)
    monkeypatch.setattr(BatchProcessor, 'post_process', lambda self: None)   # no batch outputs

    BatchProcessor(param_file, str(input_dir), str(out_dir), batch_size=5, batch_num=0,
                   resume=False, ignore_large=False).run()
    assert not (out_dir / 'stale.txt').exists() and not (out_dir / 'StaleDir').exists()
    logs = sorted(os.listdir(out_dir / logs_name))
    assert 'old.log' in logs and len(logs) == 2          # history kept, new log added
    new = [f for f in logs if f != 'old.log'][0]
    assert 'Finished in' in _read(out_dir / logs_name / new)
    assert _run_handlers() == []


def test_batch_setup_failure_opens_no_log(tmp_path):
    from cabana.batch import BatchProcessor

    input_dir, out_dir, _ = _batch_inputs(tmp_path)
    bad = tmp_path / 'bad.yml'
    bad.write_text('Configs: [unclosed')
    with pytest.raises(ValueError):
        BatchProcessor(str(bad), str(input_dir), str(out_dir), batch_size=5, batch_num=0,
                       resume=False, ignore_large=False)
    assert _run_handlers() == []
    assert not (out_dir / 'Logs').exists()
