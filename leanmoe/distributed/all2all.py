"""EP 的 token 搬运：allgather_reducescatter 与 nccl_p2p 两种实现。"""

from __future__ import annotations

import torch
import torch.distributed as dist

from leanmoe.logger import get_logger

logger = get_logger("all2all")


def _all_gather_variable(
    tensor: torch.Tensor, world_size: int, group
) -> torch.Tensor:
    """把各 rank 形状相同的 tensor 沿第 0 维拼接。"""
    if world_size == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=0)


def _reduce_scatter_variable(tensor: torch.Tensor, world_size: int, group) -> torch.Tensor:
    """沿第 0 维切成 world_size 份，各 rank 取自己那份并求和。"""
    if world_size == 1:
        return tensor
    if tensor.shape[0] % world_size != 0:
        raise ValueError(
            f"reduce_scatter 要求第 0 维能被 {world_size} 整除，收到 {tensor.shape[0]}"
        )
    output = torch.empty_like(tensor[: tensor.shape[0] // world_size])
    dist.reduce_scatter_tensor(output, tensor.contiguous(), group=group)
    return output


def allgather_reducescatter_dispatch(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AgRs dispatch：三个张量分别 all-gather，返回完整 token 集合。"""
    world_size = dist.get_world_size(group)
    hidden_all = _all_gather_variable(hidden, world_size, group)
    ids_all = _all_gather_variable(topk_ids, world_size, group)
    weights_all = _all_gather_variable(topk_weights, world_size, group)
    return hidden_all, ids_all, weights_all


def allgather_reducescatter_combine(partial: torch.Tensor, world_size: int, group) -> torch.Tensor:
    """AgRs combine：reduce-scatter 回各 rank。"""
    return _reduce_scatter_variable(partial, world_size, group)
