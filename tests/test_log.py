"""Tests for the ``cabana.log`` facade."""

import os
import re
import glob
import logging

import yaml
import pytest

from cabana.log import Log, LOGGER_NAME


@pytest.fixture
def fresh_log():
    """Leave no file handler behind for other tests."""
    yield Log
    Log.close_log_file()


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def test_logger_is_named_and_root_untouched(fresh_log, tmp_path):
    assert Log.logger.name == LOGGER_NAME
    Log.init_log_path(str(tmp_path))
    assert Log._file_handler not in logging.getLogger().handlers
    logging.getLogger('PIL').warning('third-party warning')   # WARNING: not dropped by level
    Log.logger.info('own message')
    text = _read(Log.log_name)
    assert 'own message' in text
    assert 'third-party warning' not in text


def test_console_lines_are_not_duplicated_by_root_handlers():
    """A root handler (logging.basicConfig in a script) must not repeat our lines."""
    assert Log.logger.propagate is False
    seen = []
    root_handler = logging.Handler()
    root_handler.emit = seen.append
    logging.getLogger().addHandler(root_handler)
    try:
        Log.logger.warning('once')
    finally:
        logging.getLogger().removeHandler(root_handler)
    assert seen == []


def test_reinit_replaces_file_handler(fresh_log, tmp_path):
    Log.init_log_path(str(tmp_path / 'a'))
    first = Log.log_name
    Log.logger.info('first run')
    Log.init_log_path(str(tmp_path / 'b'))
    Log.logger.info('second run')
    file_handlers = [h for h in Log.logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert 'second run' not in _read(first)
    assert 'second run' in _read(Log.log_name)


def test_file_is_plain_text_with_timestamps(fresh_log, tmp_path):
    Log.init_log_path(str(tmp_path))
    Log.logger.warning('careful')
    Log.logger.debug('details')
    text = _read(Log.log_name)
    assert '\x1b[' not in text
    assert re.search(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[WARNING\] cabana: careful$', text, re.M)
    assert '[DEBUG] cabana: details' in text


def test_module_loggers_reach_the_file(fresh_log, tmp_path):
    Log.init_log_path(str(tmp_path))
    logging.getLogger('cabana.scores').info('from scores')
    assert '[INFO] cabana.scores: from scores' in _read(Log.log_name)


def test_parameters_are_logged_at_debug(fresh_log, tmp_path):
    params = tmp_path / 'p.yml'
    yaml.safe_dump({'Detection': {'Min Line Width': 5}}, open(params, 'w'))
    Log.init_log_path(str(tmp_path / 'logs'))
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    Log.logger.addHandler(handler)
    try:
        Log.log_parameters(str(params))
    finally:
        Log.logger.removeHandler(handler)
    assert records and all(r.levelno == logging.DEBUG for r in records)
    assert 'Min Line Width: 5' in _read(Log.log_name)


def test_close_log_file_is_idempotent(fresh_log, tmp_path):
    Log.init_log_path(str(tmp_path))
    Log.close_log_file()
    Log.close_log_file()
    assert not any(isinstance(h, logging.FileHandler) for h in Log.logger.handlers)


def test_run_log_leaves_session_log_in_place(fresh_log, tmp_path):
    """A script's own log (init_log_path) keeps receiving messages after a run."""
    Log.init_log_path(str(tmp_path / 'session'))
    with Log.run_log(str(tmp_path / 'run')) as run:
        Log.logger.info('during run')
    Log.logger.info('after run')
    session = _read(Log.log_name)
    assert 'during run' in session and 'after run' in session
    run_text = _read(run.path)
    assert 'during run' in run_text and 'after run' not in run_text
    assert [h for h in Log.logger.handlers if isinstance(h, logging.FileHandler)] == [Log._file_handler]


def test_run_log_records_exception_and_closes(tmp_path):
    with pytest.raises(RuntimeError):
        with Log.run_log(str(tmp_path)) as run:
            raise RuntimeError('boom')
    text = _read(run.path)
    assert 'Run failed' in text and 'RuntimeError: boom' in text
    assert not any(isinstance(h, logging.FileHandler) for h in Log.logger.handlers)
    run.close()        # idempotent


def test_run_logs_in_the_same_second_get_distinct_files(tmp_path):
    a = Log.run_log(str(tmp_path))
    b = Log.run_log(str(tmp_path))
    try:
        assert a.path != b.path
        assert len(glob.glob(str(tmp_path / '*.log'))) == 2
    finally:
        a.close()
        b.close()


class _FakeBatchCabana:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        pass


def test_batch_log_lives_in_output_and_survives_a_new_run(fresh_log, tmp_path, monkeypatch):
    import numpy as np
    import cv2
    import cabana.batch as batch_mod
    from cabana.batch import BatchProcessor

    input_dir = tmp_path / 'in'
    input_dir.mkdir()
    cv2.imwrite(str(input_dir / 'a.png'), np.full((40, 40, 3), 200, np.uint8))
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'stale.txt').write_text('old result')
    (out_dir / 'StaleDir').mkdir()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(root, 'cabana', 'default_params.yml')
    monkeypatch.setattr(batch_mod, 'BatchCabana', _FakeBatchCabana)

    bp = BatchProcessor(param_file, str(input_dir), str(out_dir), batch_size=5, batch_num=0,
                        resume=False, ignore_large=False)
    assert os.path.dirname(bp.run_log.path) == os.path.join(str(out_dir), 'Logs').replace('\\', '/')
    bp.process()
    assert not (out_dir / 'stale.txt').exists() and not (out_dir / 'StaleDir').exists()
    logs = glob.glob(str(out_dir / 'Logs' / '*.log'))
    assert len(logs) == 1
    Log.logger.info('still writing')
    assert 'still writing' in _read(logs[0])
    assert 'Min Line Width' in _read(logs[0])      # parameters recorded in the file
    bp.run_log.close()


def test_batch_setup_failure_closes_its_log(tmp_path):
    """A bad parameter file must not leave the run's log attached to the logger."""
    import numpy as np
    import cv2
    from cabana.batch import BatchProcessor

    input_dir = tmp_path / 'in'
    input_dir.mkdir()
    cv2.imwrite(str(input_dir / 'a.png'), np.full((40, 40, 3), 200, np.uint8))
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    bad = tmp_path / 'bad.yml'
    bad.write_text('Configs: [unclosed')
    with pytest.raises(ValueError):
        BatchProcessor(str(bad), str(input_dir), str(out_dir), batch_size=5, batch_num=0,
                       resume=False, ignore_large=False)
    assert not any(isinstance(h, logging.FileHandler) for h in Log.logger.handlers)
    logs = glob.glob(str(out_dir / 'Logs' / '*.log'))
    assert len(logs) == 1 and 'Batch setup failed' in _read(logs[0])
