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
    logging.getLogger('PIL').debug('third-party debug')
    Log.logger.info('own message')
    text = _read(Log.log_name)
    assert 'own message' in text
    assert 'third-party debug' not in text


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
    assert os.path.dirname(Log.log_name) == os.path.join(str(out_dir), 'Logs').replace('\\', '/')
    bp.process()
    assert not (out_dir / 'stale.txt').exists() and not (out_dir / 'StaleDir').exists()
    logs = glob.glob(str(out_dir / 'Logs' / '*.log'))
    assert len(logs) == 1
    Log.logger.info('still writing')
    assert 'still writing' in _read(logs[0])
    assert 'Min Line Width' in _read(logs[0])      # parameters recorded in the file
