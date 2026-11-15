"""RoPE：标准旋转位置编码，支持简单的 linear / yarn 缩放。"""

from __future__ import annotations

import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int | None = None,
        max_position_embeddings: int = 32768,
        base: float = 1_000_000.0,
        dtype: torch.dtype = torch.bfloat16,
        scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim or head_dim
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim 必须是偶数")
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling = scaling or {}

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 预计算到 max_model_len，避免每步重算
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._built_len = 0
        self.dtype = dtype

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if self._cos is not None and self._cos.shape[0] >= seq_len and self._cos.device == device:
            return
        length = max(seq_len, self.max_position_embeddings)
        t = torch.arange(length, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos().to(self.dtype)
        self._sin = emb.sin().to(self.dtype)
        self._built_len = length

    def get_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """返回整张 cos/sin 表（[max_pos, D]），由 kernel 按 position 索引。"""
        if self._cos is None or self._cos.device != device:
            self._build_cache(self.max_position_embeddings, device)
        return self._cos, self._sin
