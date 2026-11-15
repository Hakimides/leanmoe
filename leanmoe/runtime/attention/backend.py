"""Attention 门面：保持原有导入路径可用，实际实现已拆分到子模块。"""

from leanmoe.runtime.attention.backends import (
    FlashAttnBackend,
    SDPABackend,
    TritonPagedDecodeBackend,
    gather_kv,
)
from leanmoe.runtime.attention.kernels import (
    _gqa_combine_kernel,
    _paged_decode_gqa_split_kernel,
    _paged_decode_kernel,
    _paged_decode_split_kernel,
    _paged_prefill_kernel,
    _reshape_and_cache_kernel,
    _split_combine_kernel,
    write_to_kv_cache,
)
from leanmoe.runtime.attention.metadata import AttentionBackend, AttentionMetadata
from leanmoe.runtime.attention.selector import get_attn_backend

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "FlashAttnBackend",
    "SDPABackend",
    "TritonPagedDecodeBackend",
    "gather_kv",
    "get_attn_backend",
    "write_to_kv_cache",
]
