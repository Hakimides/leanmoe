"""Attention 后端选择器。"""

from __future__ import annotations

import os

from leanmoe.runtime.attention.backends import (
    TritonPagedDecodeBackend,
    FlashAttnBackend,
    _SDPA_FALLBACK,
)
from leanmoe.runtime.attention.metadata import AttentionBackend

_SDPA = _SDPA_FALLBACK
_FA2 = FlashAttnBackend()
_TP = TritonPagedDecodeBackend()


def get_attn_backend(
    backend: str = "auto",
    capturing: bool = False,
) -> AttentionBackend:
    """按环境变量与捕获状态选择 attention 后端。"""
    backend = os.environ.get("LEANMOE_ATTN_BACKEND", backend)
    if capturing:
        if _FA2.available:
            return _FA2
        return _TP if _TP.available else _SDPA
    if backend == "sdpa":
        return _SDPA
    if backend == "flash_attn":
        return _FA2 if _FA2.available else _SDPA
    if backend == "triton_paged":
        return _TP if _TP.available else _SDPA
    if backend == "auto":
        if _FA2.available:
            return _FA2
        if _TP.available:
            return _TP
        return _SDPA
    raise ValueError(f"未知的 attention 后端: {backend}")
