"""MoERunner：路由 → dispatch → 本地专家 → combine 的编排。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.distributed.all2all import (
    allgather_reducescatter_combine,
    allgather_reducescatter_dispatch,
)
from leanmoe.distributed.parallel_state import get_ep_group, get_tp_group
from leanmoe.logger import get_logger
from leanmoe.model_executor.layers.fused_moe.config import FusedMoEConfig
from leanmoe.model_executor.layers.fused_moe.experts import RoutedExperts
from leanmoe.model_executor.layers.fused_moe.router import Router, RouterOutput

logger = get_logger("moe.runner")


class MoERunner(nn.Module):
    def __init__(
        self,
        router: Router,
        experts: RoutedExperts,
        config: FusedMoEConfig,
        dp_size: int = 1,
        pp_size: int = 1,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.config = config
        self.dp_size = dp_size
        self.pp_size = pp_size
        # 纯 TP+EP（dp=pp=1）不走 all-to-all，走 NoDPEP + all-reduce
        self.use_all2all = config.use_ep and (dp_size > 1 or pp_size > 1)
        self.ep_rank = get_ep_group().rank_in_group if config.use_ep else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_out: RouterOutput = self.router(hidden_states)
        topk_ids = router_out.topk_ids
        topk_weights = router_out.topk_weights

        if not self.config.use_ep:
            out = self.experts(hidden_states, topk_ids, topk_weights)
            if self.config.tp_size > 1:
                out = get_tp_group().all_reduce(out)
            return out

        if self.use_all2all:
            return self._forward_all2all(hidden_states, topk_ids, topk_weights)
        return self._forward_no_dp_ep(hidden_states, topk_ids, topk_weights)

    # ---------- EP：AgRs ----------

    def _forward_all2all(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        group = get_ep_group()
        ep_size = group.world_size
        num_local = self.config.num_local_experts
        ep_rank = group.rank_in_group

        hidden_all, ids_all, weights_all = allgather_reducescatter_dispatch(
            hidden_states, topk_ids, topk_weights, group.group
        )

        # 只保留落在本 rank 专家区间内的 (token, expert) 对
        local_ids = ids_all - ep_rank * num_local
        valid = (local_ids >= 0) & (local_ids < num_local)
        safe_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
        masked_weights = torch.where(
            valid, weights_all, torch.zeros_like(weights_all)
        )
        partial = self.experts(hidden_all, safe_ids, masked_weights)
        out = allgather_reducescatter_combine(partial, ep_size, group.group)
        return out

    # ---------- EP：NoDPEP ----------

    def _forward_no_dp_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        group = get_ep_group()
        num_local = self.config.num_local_experts
        ep_rank = group.rank_in_group

        local_ids = topk_ids - ep_rank * num_local
        valid = (local_ids >= 0) & (local_ids < num_local)
        safe_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
        masked_weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))

        partial = self.experts(hidden_states, safe_ids, masked_weights)
        if group.world_size > 1:
            partial = group.all_reduce(partial)
        return partial

    # ---------- 路由观测 ----------

    def pop_routed_experts(self) -> torch.Tensor | None:
        """返回本层最近一次的 logical top-k 路由（[T, K]），未开启时为 None。"""
        return self.router.last_topk_ids


def create_moe_runner(
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    norm_topk_prob: bool,
    config: FusedMoEConfig,
    params_dtype: torch.dtype,
    dp_size: int = 1,
    pp_size: int = 1,
    router_quant_method=None,
    experts_quant_method=None,
) -> MoERunner:
    router = Router(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        params_dtype=params_dtype,
        quant_method=router_quant_method,
    )
    experts = RoutedExperts(
        num_local_experts=config.num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=config.intermediate_per_partition,
        params_dtype=params_dtype,
    )
    # 量化方法在这里挂到 experts 上；未实现时保持 bf16 参考路径
    if experts_quant_method is not None:
        experts.quant_method = experts_quant_method
    return MoERunner(router, experts, config, dp_size=dp_size, pp_size=pp_size)
