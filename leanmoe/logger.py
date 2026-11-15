"""日志：统一前缀与一次性告警去重。"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger("leanmoe")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _ensure_configured()
    return logging.getLogger(f"leanmoe.{name}")


class _Once:
    """同一类告警只打印一次，避免每步刷屏。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def warn(self, logger: logging.Logger, key: str, message: str) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        logger.warning(message)

    def info(self, logger: logging.Logger, key: str, message: str) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        logger.info(message)


once = _Once()
