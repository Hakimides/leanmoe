"""RoutedExperts：专家权重与本地专家计算。"""

from __future__ import annotations

import os

import torch
from torch import nn

from leanmoe.logger import get_logger, once
from leanmoe.model_executor.layers.fused_moe.kernels import (
    grouped_gemm,
    grouped_gemm_silu,
    moe_align_block_size_fast,
    moe_combine,
    moe_weighted_scatter,
)
from leanmoe.model_executor.layers.silu_and_mul import silu_and_mul

logger = get_logger("moe.experts")


class RoutedExperts(nn.Module):
    """本地专家权重。"""

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.w13 = nn.Parameter(
            torch.empty(num_local_experts, hidden_size, 2 * intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.w2 = nn.Parameter(
            torch.empty(num_local_experts, intermediate_size, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states: [T, H]；topk_ids/weights: [T, K]（本地专家 id）。"""
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(hidden_states.shape)
        return self._forward_tuned(hidden_states, topk_ids, topk_weights)

    def _forward_tuned(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """模仿 vLLM：w13 融合 SiLU + 它的默认 tile 配置（decode M=16 / prefill 大连批）。"""
        num_tokens = hidden_states.shape[0]
        # prefill 走移植版（vLLM 布局 + 指针式 GEMM 内核，实测大行数更快）
        if (
            os.environ.get("LEANMOE_PORT_MOE", "1") == "1"
            and num_tokens * topk_ids.shape[1] > 1024
        ):
            out = self._forward_port(hidden_states, topk_ids, topk_weights)
            if out is not None:
                return out
        sorted_ids, offsets, flat_indices, inv = moe_align_block_size_fast(
            topk_ids, self.num_local_experts
        )
        n_rows = sorted_ids.numel()
        if n_rows > 1024:
            # prefill：融合 SiLU 的 64/128/32/warps8（实测最优；BM=128 会寄存器溢出）
            opts = {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 8}
            inputs = hidden_states.index_select(0, sorted_ids)
            intermediate = grouped_gemm_silu(inputs, self.w13, offsets, **opts)
        else:
            # decode：行索引进 kernel，省掉每层一次 index_select
            opts = {"block_m": 16, "block_n": 64, "block_k": 128, "num_warps": 4}
            intermediate = grouped_gemm_silu(
                hidden_states, self.w13, offsets, row_idx=sorted_ids, **opts
            )
        expert_out = grouped_gemm(intermediate, self.w2, offsets, **opts)

        # 优先用非原子融合求和（prefill 的 fp32 原子散射是主要瓶颈）
        if inv is not None:
            combined = moe_combine(expert_out, inv, topk_weights, num_tokens)
            if combined is not None:
                return combined

        if expert_out.shape[0] > 1024:
            return moe_weighted_scatter(
                expert_out,
                sorted_ids,
                flat_indices,
                topk_weights.reshape(-1),
                num_tokens,
            )

        flat_weights = topk_weights.reshape(-1)
        valid = flat_indices >= 0
        gathered_w = flat_weights.index_select(0, flat_indices.clamp(min=0))
        weights = torch.where(
            valid,
            gathered_w,
            torch.zeros_like(flat_weights[:1]).expand_as(flat_indices),
        )
        total_rows = sorted_ids.numel()
        scatter_ids = torch.where(
            valid,
            sorted_ids,
            num_tokens + torch.arange(total_rows, device=hidden_states.device),
        )
        output = hidden_states.new_zeros(
            (num_tokens + total_rows, hidden_states.shape[-1])
        )
        output.index_add_(0, scatter_ids, expert_out * weights.unsqueeze(-1))
        return output[:num_tokens]

    def _forward_port(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """本地移植版（与本地 GEMM 共用 [E, K, N] 权重布局），失败时返回 None 回退。"""
        try:
            from leanmoe.model_executor.layers.fused_moe.triton_moe_port import (
                fused_experts_port,
            )
        except Exception as exc:
            once.warn(logger, "port_moe_import", f"移植版 MoE 不可用，回退: {exc}")
            os.environ["LEANMOE_PORT_MOE"] = "0"
            return None
        try:
            return fused_experts_port(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, silu_and_mul,
            )
        except Exception as exc:
            once.warn(logger, "port_moe_fail", f"移植版 MoE 失败，回退: {exc}")
            os.environ["LEANMOE_PORT_MOE"] = "0"
            return None
