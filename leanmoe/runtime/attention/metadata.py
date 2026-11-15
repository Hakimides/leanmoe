"""Attention 元数据与后端接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class AttentionMetadata:
    """一次 forward 的注意力元数据。"""

    # 请求数与各段长度
    num_seqs: int
    num_prefill_seqs: int
    num_decode_seqs: int
    num_prefill_tokens: int
    num_decode_tokens: int
    # [num_seqs + 1] 的 query 累计长度（扁平 token 布局）
    query_start_loc: torch.Tensor
    # [num_seqs] 每序列本步结束后的 KV 长度
    seq_lens: torch.Tensor
    # [num_seqs, max_blocks] 逻辑块表
    block_tables: torch.Tensor
    # [num_tokens] 新 token 写入的扁平槽位
    slot_mapping: torch.Tensor
    # 纯 decode 时为 True，可走批量路径
    is_pure_decode: bool = False
    # 最大 query 长度
    max_query_len: int = 1
    # 所有请求都是首块且等长时，记录该长度，用于批量 prefill
    uniform_prefill_len: int | None = None
    # paged decode 的 KV 上界与 q 的累计长度
    max_seqlen_k: int = 0
    decode_cu_seqlens: torch.Tensor | None = None
    # 是否为 CUDA Graph 捕获/回放
    is_capturing: bool = False

    # 以下字段由后端按需填充
    extra: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


class AttentionBackend(Protocol):
    name: str

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        ...


