"""并行状态：TP / DP / PP / EP 组的构建与集合通信封装。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from leanmoe.logger import get_logger

logger = get_logger("distributed")


@dataclass
class ParallelState:
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def dp_rank(self) -> int:
        return (self.rank // self.tp_size) % self.dp_size

    @property
    def pp_rank(self) -> int:
        return self.rank // (self.tp_size * self.dp_size)

    @property
    def ep_size(self) -> int:
        return self.tp_size * self.dp_size

    @property
    def ep_rank(self) -> int:
        return self.dp_rank * self.tp_size + self.tp_rank

    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1


_PARALLEL_STATE = ParallelState()


def get_parallel_state() -> ParallelState:
    return _PARALLEL_STATE


def init_distributed_environment(
    world_size: int,
    rank: int,
    local_rank: int,
    backend: str = "nccl",
    distributed_init_method: str | None = None,
) -> None:
    """初始化进程组。单卡时也走这里，保持调用路径一致。"""
    global _PARALLEL_STATE
    if world_size == 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        _PARALLEL_STATE = ParallelState(1, 1, 1, 0, local_rank, 1)
        return

    if not dist.is_initialized():
        if distributed_init_method is None:
            distributed_init_method = f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:" \
                                     f"{os.environ.get('MASTER_PORT', '29500')}"
        dist.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    logger.info("distributed 初始化完成：world_size=%d rank=%d", world_size, rank)


class GroupCoordinator:
    """一个通信组的封装。

    所有方法都接受可选 group，未初始化分布式时退化为本地 no-op。
    """

    def __init__(self, ranks: list[int], group_name: str) -> None:
        self.ranks = ranks
        self.group_name = group_name
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(dist.get_rank()) if dist.is_initialized() else 0
        self.group = None
        # 自研 CUDA IPC all-reduce（2 卡 NVLink 下图内约 6 µs/次，NCCL 约 50 µs）
        self._custom_ar = None
        if dist.is_initialized() and self.world_size > 1:
            self.group = dist.new_group(ranks)
            if (
                group_name == "tp"
                and self.world_size == 2
                and os.environ.get("LEANMOE_CUSTOM_AR", "1") == "1"
            ):
                self._init_custom_ar()

    def _init_custom_ar(self) -> None:
        try:
            from leanmoe.distributed.custom_ar import IPCAllReduce
        except Exception as exc:  # 扩展不可用时保持 NCCL
            logger.info("自研 IPCAllReduce 不可用，使用 NCCL: %s", exc)
            return
        try:
            # IPC 句柄交换走 gloo 组；数据面走 CUDA IPC
            cpu_group = dist.new_group(self.ranks, backend="gloo")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._custom_ar = IPCAllReduce(
                cpu_group,
                self.rank_in_group,
                self.world_size,
                device=torch.device("cuda", local_rank),
            )
            logger.info("启用自研 IPCAllReduce（2 卡 CUDA IPC）")
        except Exception as exc:
            logger.warning("IPCAllReduce 初始化失败，保持 NCCL: %s", exc)
            self._custom_ar = None

    @contextmanager
    def capture_custom_all_reduce(self):
        """barrier 已融进 sum kernel，图捕获期无需特殊处理。"""
        yield

    # ---------- 基础 ----------

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        # 调试开关：跳过集合通信，用于量化 all-reduce 的端到端代价（结果不可用）
        """组内 all-reduce，小消息优先自研 IPC。"""
        if os.environ.get("LEANMOE_SKIP_AR") == "1":
            return tensor
        # 小消息走自研 IPC（可被 CUDA Graph 捕获）；超阈值或非 bf16 返回 None，回落 NCCL
        if self._custom_ar is not None:
            out = self._custom_ar.all_reduce(tensor)
            if out is not None:
                return out
        if self.world_size > 1:
            dist.all_reduce(tensor, group=self.group)
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        # 同上：调试开关同时跳过 all_gather，否则它会成为图捕获期第一个 NCCL 调用而报错
        """组内 all-gather 并拼接。"""
        if os.environ.get("LEANMOE_SKIP_AR") == "1":
            return tensor
        if self.world_size == 1:
            return tensor
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor.contiguous(), group=self.group)
        return torch.cat(parts, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """组内 reduce-scatter。"""
        if self.world_size == 1:
            return tensor
        chunks = tensor.chunk(self.world_size, dim=dim)
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter_tensor(output, tensor.contiguous(), group=self.group)
        return output

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """组内广播。"""
        if self.world_size > 1:
            dist.broadcast(tensor, src=self.ranks[src], group=self.group)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        """点对点发送。"""
        if self.world_size > 1:
            dist.send(tensor.contiguous(), dst=self.ranks[dst], group=self.group)

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """点对点接收。"""
        if self.world_size > 1:
            dist.recv(tensor, src=self.ranks[src], group=self.group)
        return tensor

    def isend(self, tensor: torch.Tensor, dst: int):
        """异步发送。"""
        if self.world_size == 1:
            return None
        return dist.isend(tensor.contiguous(), dst=self.ranks[dst], group=self.group)

    def irecv(self, tensor: torch.Tensor, src: int):
        """异步接收。"""
        if self.world_size == 1:
            return None
        return dist.irecv(tensor, src=self.ranks[src], group=self.group)

    def barrier(self) -> None:
        """组内 barrier。"""
        if self.world_size > 1:
            dist.barrier(group=self.group)


# 全局通信组，initialize_model_parallel 后可用
_TP_GROUP: GroupCoordinator | None = None
_PP_GROUP: GroupCoordinator | None = None
_DP_GROUP: GroupCoordinator | None = None
_EP_GROUP: GroupCoordinator | None = None


def initialize_model_parallel(
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> ParallelState:
    """构建 TP / PP / DP / EP 通信组，返回并行状态。"""
    global _PARALLEL_STATE, _TP_GROUP, _PP_GROUP, _DP_GROUP, _EP_GROUP

    if not dist.is_initialized():
        _PARALLEL_STATE = ParallelState(
            tensor_parallel_size, data_parallel_size, pipeline_parallel_size, 0, 0, 1
        )
        # 单卡也建立 world_size=1 的组，保证模型构造时 get_*_group 可用
        _TP_GROUP = GroupCoordinator([0], "tp")
        _PP_GROUP = GroupCoordinator([0], "pp")
        _DP_GROUP = GroupCoordinator([0], "dp")
        _EP_GROUP = GroupCoordinator([0], "ep")
        return _PARALLEL_STATE

    world_size = dist.get_world_size()
    expected = tensor_parallel_size * data_parallel_size * pipeline_parallel_size
    if world_size != expected:
        raise ValueError(
            f"world_size={world_size} 与 tp×dp×pp={expected} 不一致；"
            f"请用 torchrun --nproc-per-node={expected} 启动"
        )
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    _PARALLEL_STATE = ParallelState(
        tensor_parallel_size, data_parallel_size, pipeline_parallel_size, rank, local_rank, world_size
    )

    tp_ranks, pp_ranks, dp_ranks, ep_ranks = [], [], [], []
    for pp in range(pipeline_parallel_size):
        for dp in range(data_parallel_size):
            for tp in range(tensor_parallel_size):
                global_rank = pp * (data_parallel_size * tensor_parallel_size) \
                              + dp * tensor_parallel_size + tp
                tp_ranks.append(global_rank)
                pp_ranks.append(global_rank)
                dp_ranks.append(global_rank)
                ep_ranks.append(global_rank)

    def _slice(full: list[int], size: int, index: int) -> list[int]:
        return full[index * size : (index + 1) * size]

    state = _PARALLEL_STATE
    tp_group_ranks = _slice(tp_ranks, tensor_parallel_size,
                            (state.pp_rank * data_parallel_size + state.dp_rank))
    pp_group_ranks = [pp * (data_parallel_size * tensor_parallel_size)
                      + state.dp_rank * tensor_parallel_size + state.tp_rank
                      for pp in range(pipeline_parallel_size)]
    dp_group_ranks = [state.pp_rank * (data_parallel_size * tensor_parallel_size)
                      + dp * tensor_parallel_size + state.tp_rank
                      for dp in range(data_parallel_size)]
    ep_group_ranks = [state.pp_rank * (data_parallel_size * tensor_parallel_size)
                      + dp * tensor_parallel_size + tp
                      for dp in range(data_parallel_size)
                      for tp in range(tensor_parallel_size)]

    _TP_GROUP = GroupCoordinator(tp_group_ranks, "tp")
    _PP_GROUP = GroupCoordinator(pp_group_ranks, "pp")
    _DP_GROUP = GroupCoordinator(dp_group_ranks, "dp")
    _EP_GROUP = GroupCoordinator(ep_group_ranks, "ep")

    logger.info(
        "并行组就绪：tp=%s pp=%s dp=%s ep=%s",
        tp_group_ranks, pp_group_ranks, dp_group_ranks, ep_group_ranks,
    )
    return _PARALLEL_STATE


def get_tp_group() -> GroupCoordinator:
    """返回 TP 进程组。"""
    if _TP_GROUP is None:
        raise RuntimeError("并行组未初始化，先调用 initialize_model_parallel")
    return _TP_GROUP


def get_pp_group() -> GroupCoordinator:
    """返回 PP 进程组。"""
    if _PP_GROUP is None:
        raise RuntimeError("并行组未初始化")
    return _PP_GROUP


def get_dp_group() -> GroupCoordinator:
    """返回 DP 进程组。"""
    if _DP_GROUP is None:
        raise RuntimeError("并行组未初始化")
    return _DP_GROUP


def get_ep_group() -> GroupCoordinator:
    """返回 EP 进程组。"""
    if _EP_GROUP is None:
        raise RuntimeError("并行组未初始化")
    return _EP_GROUP


def destroy_model_parallel() -> None:
    """销毁进程组并复位全局状态。"""
    global _TP_GROUP, _PP_GROUP, _DP_GROUP, _EP_GROUP
    _TP_GROUP = _PP_GROUP = _DP_GROUP = _EP_GROUP = None
    if dist.is_initialized():
        dist.destroy_process_group()
