"""2 卡 CUDA IPC all-reduce（本项目自研，不依赖 vllm）。"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

# 超过 8 MB 或非 bf16 回落到 NCCL：大消息 NCCL 更合适
_MAX_BYTES = 8 * 1024 * 1024
_ext = None
_ext_error: str | None = None


def _load_ext():
    global _ext, _ext_error
    if _ext is not None or _ext_error is not None:
        return _ext
    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("csrc") / "custom_ar.cu"
        _ext = load(
            name="leanmoe_custom_ar",
            sources=[str(src)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # 编译失败时上层回退 NCCL
        _ext_error = str(exc)
        return None
    return _ext


def _peer_ptr(ext, tensor: torch.Tensor, cpu_group, peer: int) -> int:
    handle = torch.empty(64, dtype=torch.uint8)
    ext.get_mem_handle(tensor.data_ptr(), handle.data_ptr())
    objs: list[bytes | None] = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(objs, bytes(handle.tolist()), group=cpu_group)
    peer_bytes = torch.tensor(list(objs[peer]), dtype=torch.uint8)
    return int(ext.open_mem_handle(peer_bytes.data_ptr()))


class IPCAllReduce:
    """仅支持 2 卡、bf16、单次 <= 8 MB。"""

    def __init__(self, cpu_group, rank_in_group: int, world_size: int,
                 device: torch.device | None = None) -> None:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"custom_ar 扩展不可用: {_ext_error}")
        if world_size != 2:
            raise RuntimeError("IPCAllReduce 只支持 2 卡")
        self.ext = ext
        self.rank = rank_in_group
        self.peer = 1 - rank_in_group
        if device is None:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        self.send = torch.empty(_MAX_BYTES // 2, dtype=torch.bfloat16, device=device)
        self.ctr = torch.zeros(1, dtype=torch.int64, device=device)

        self.peer_ptr = _peer_ptr(ext, self.send, cpu_group, self.peer)
        self.peer_ctr = _peer_ptr(ext, self.ctr, cpu_group, self.peer)

        if os.environ.get("LEANMOE_AR_DEBUG") == "1":
            print(f"[custom_ar] rank={self.rank} 初始化完成", flush=True)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor | None:
        n = tensor.numel()
        if n * tensor.element_size() > _MAX_BYTES or tensor.dtype != torch.bfloat16:
            return None
        self.send[:n].copy_(tensor.reshape(-1))
        out = torch.empty_like(tensor)
        self.ext.sum2(
            tensor.data_ptr(),
            self.peer_ptr,
            out.data_ptr(),
            n,
            self.ctr.data_ptr(),
            self.peer_ctr,
        )
        return out
