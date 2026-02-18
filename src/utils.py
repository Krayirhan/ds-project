"""
utils.py

Genel yardımcı fonksiyonlar.

Neden ayrı dosya?
- Logging standardı tüm projede aynı olsun.
- Seed ayarı tek yerden yönetilsin.
- Kod tekrarını azaltır.
"""

import hashlib
import json
import logging
import os
import random
import numpy as np


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = getattr(record, "request_id")
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Logger oluşturur.

    Neden handler kontrolü var?
    - Aynı logger tekrar oluşturulursa duplicate log basılabilir.

    LOG_LEVEL env var:
    - DEBUG, INFO, WARNING, ERROR, CRITICAL değerlerini kabul eder.
    - Yalnızca handler kurulurken okunur; her çağrıda override edilmez.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        use_json = os.getenv("LOG_FORMAT", "plain").lower() == "json"
        if use_json:
            formatter = JsonLogFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # LOG_LEVEL env var'dan oku; bulunamazsa caller'ın istediği level'ı kullan
        env_level_str = os.getenv("LOG_LEVEL", "").upper()
        resolved_level = getattr(logging, env_level_str, None) or level
        logger.setLevel(resolved_level)
    return logger


def set_seed(seed: int) -> None:
    """
    Reproducibility için seed set eder.

    Neden önemli?
    - Split shuffling
    - CV fold randomization
    - Bazı model iç rastgelelikleri
    sonuçları etkiler.
    """
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
