"""Router：gate → top-k → 可选 renormalize。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from leanmoe.model_executor.layers.fused_topk import topk_softmax
from leanmoe.model_executor.layers.linear import ReplicatedLinear


@dataclass
class RouterOutput:
    topk_ids: torch.Tensor       # [T, K] logical expert id
    topk_weights: torch.Tensor   # [T, K]
    # [T, K] 原始 logits 最大值处的 expert（可选，用于日志）
    logits: torch.Tensor | None = None


class Router(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
        quant_method=None,
    ) -> None:
        super().__init__()
        self.gate = ReplicatedLinear(
            hidden_size, num_experts, bias=False,
            quant_method=quant_method, params_dtype=params_dtype,
        )
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = num_experts
        # 调试开关：记录本步路由结果
        self.record_routing = False
        self.last_topk_ids: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        logits = self.gate(hidden_states)
        # 融合 softmax + top-k（含可选 renormalize），省 2-3 个 kernel
        topk_ids, topk_weights = topk_softmax(
            logits, self.top_k, self.norm_topk_prob
        )
        if self.record_routing:
            self.last_topk_ids = topk_ids.detach()
        return RouterOutput(topk_ids=topk_ids, topk_weights=topk_weights)

