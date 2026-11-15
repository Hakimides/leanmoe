"""集合通信算子封装。"""

from __future__ import annotations

import torch

from leanmoe.distributed.parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(tensor)


def tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return get_tp_group().all_gather(tensor, dim=dim)
