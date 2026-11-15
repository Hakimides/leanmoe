"""通用工具：整除、时钟、哈希。"""

from __future__ import annotations

import hashlib
import time


def cdiv(a: int, b: int) -> int:
    """向上取整除法。"""
    return -(a // -b)


def sha256_hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def now() -> float:
    """单调时钟，用于 TTFT/TPOT 统计。"""
    return time.perf_counter()
