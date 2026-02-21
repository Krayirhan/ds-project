from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid

import numpy as np

from src.utils import JsonLogFormatter, get_logger, set_seed, sha256_file


def test_json_log_formatter_includes_request_id():
    formatter = JsonLogFormatter()
    rec = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rec.request_id = "rid-1"
    payload = json.loads(formatter.format(rec))
    assert payload["logger"] == "test"
    assert payload["message"] == "hello"
    assert payload["request_id"] == "rid-1"


def test_get_logger_plain_and_json_modes(monkeypatch):
    name_plain = f"logger_plain_{uuid.uuid4().hex}"
    monkeypatch.setenv("LOG_FORMAT", "plain")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    lg_plain = get_logger(name_plain)
    assert lg_plain.level == logging.DEBUG
    assert isinstance(lg_plain.handlers[0].formatter, logging.Formatter)

    name_json = f"logger_json_{uuid.uuid4().hex}"
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    lg_json = get_logger(name_json)
    assert isinstance(lg_json.handlers[0].formatter, JsonLogFormatter)


def test_set_seed_reproducible():
    set_seed(123)
    py1 = random.random()
    np1 = float(np.random.rand())
    set_seed(123)
    py2 = random.random()
    np2 = float(np.random.rand())
    assert py1 == py2
    assert np1 == np2


def test_sha256_file(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"hello world"
    p.write_bytes(data)
    assert sha256_file(str(p)) == hashlib.sha256(data).hexdigest()
