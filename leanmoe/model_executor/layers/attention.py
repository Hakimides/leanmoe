"""模型内的 Attention 层：QKV 投影 + QK-Norm + RoPE + 后端注意力 + 输出投影。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.distributed.parallel_state import get_tp_group
from leanmoe.model_executor.layers.fused_qk_norm_rope import fused_qk_norm_rope
from leanmoe.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from leanmoe.model_executor.layers.layernorm import RMSNorm
from leanmoe.model_executor.layers.rotary_embedding import RotaryEmbedding
from leanmoe.runtime.attention.backend import AttentionMetadata, get_attn_backend


class Qwen3Attention(nn.Module):
    """QKV 投影 → QK-Norm+RoPE → paged attention → O 投影。"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary: RotaryEmbedding,
        rms_norm_eps: float,
        params_dtype: torch.dtype = torch.bfloat16,
        quant_method=None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_heads={num_heads} / num_kv_heads={num_kv_heads} 不能被 tp={tp_size} 整除"
            )
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            bias=False,
            quant_method=quant_method,
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_method=quant_method,
            params_dtype=params_dtype,
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, dtype=params_dtype)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, dtype=params_dtype)
        self.rotary = rotary
        self.backend = get_attn_backend("auto")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """一层 attention 子层的前向。"""
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 融合 QK-Norm + RoPE：每层 4 个 kernel 合成 2 个；cos/sin 用整表、kernel 内按位置索引
        cos, sin = self.rotary.get_tables(q.device)
        fused_qk_norm_rope(
            q, self.q_norm.weight, cos, sin, positions,
            self.q_norm.variance_epsilon, self.num_heads, self.head_dim,
        )
        fused_qk_norm_rope(
            k, self.k_norm.weight, cos, sin, positions,
            self.k_norm.variance_epsilon, self.num_kv_heads, self.head_dim,
        )

        backend = get_attn_backend("auto", capturing=attn_metadata.is_capturing)
        out = backend.forward(
            q, k, v, kv_cache, attn_metadata, self.num_kv_heads, self.head_dim
        )
        out = out.reshape(-1, self.num_heads * self.head_dim)
        return self.o_proj(out)
