"""PD 分离的 KV 传输连接器。"""

from __future__ import annotations

import torch

from leanmoe.config import KVTransferConfig
from leanmoe.logger import get_logger

logger = get_logger("kv_transfer")


class KVConnectorBase:
    def __init__(self, config: KVTransferConfig) -> None:
        self.config = config

    # scheduler 侧
    def request_finished(self, request) -> tuple[bool, dict | None]:
        """返回 (是否异步保存, 保存元数据)。"""
        return False, None

    # worker 侧
    def save_kv_layer(self, layer_idx: int, kv_cache: torch.Tensor, metadata: dict) -> None:
        raise NotImplementedError

    def start_load_kv(self, metadata: dict) -> None:
        raise NotImplementedError

    def wait_for_layer_load(self, layer_idx: int) -> None:
        raise NotImplementedError

    def wait_for_save(self) -> None:
        raise NotImplementedError

    def get_finished(self) -> set[str]:
        return set()


class SimpleConnector(KVConnectorBase):
    """逐层发送/接收。"""

    def __init__(self, config: KVTransferConfig) -> None:
        super().__init__(config)
        self._send_streams: list[torch.cuda.Stream] = []
        self._pending = None
        self._finished: set[str] = set()

    def request_finished(self, request) -> tuple[bool, dict | None]:
        if self.config.role != "prefill":
            return False, None
        meta = {
            "request_id": request.request_id,
            "num_blocks": len(request.block_ids),
            "block_ids": list(request.block_ids),
            "num_tokens": request.num_tokens,
        }
        return True, meta

    def save_kv_layer(self, layer_idx: int, kv_cache: torch.Tensor, metadata: dict) -> None:
        """把本层 KV 发往 decode 实例。"""
        from leanmoe.distributed.parallel_state import get_pp_group

        group = get_pp_group()
        if group.world_size <= 1:
            return
        stream = torch.cuda.Stream()
        self._send_streams.append(stream)
        with torch.cuda.stream(stream):
            group.send(kv_cache.contiguous(), dst=0)
        self._pending = (group, stream)

    def start_load_kv(self, metadata: dict) -> None:
        from leanmoe.distributed.parallel_state import get_pp_group

        self._group = get_pp_group()

    def wait_for_layer_load(self, layer_idx: int) -> None:
        # 当前实现由调用方在每层前阻塞等待；真实实现应使用独立 stream 做重叠
        return None

    def wait_for_save(self) -> None:
        if self._pending is None:
            return
        group, stream = self._pending
        torch.cuda.current_stream().wait_stream(stream)
        self._pending = None

    def get_finished(self) -> set[str]:
        finished, self._finished = self._finished, set()
        return finished


def create_kv_connector(config: KVTransferConfig) -> KVConnectorBase | None:
    if config.kv_connector == "none":
        return None
    if config.kv_connector == "simple":
        return SimpleConnector(config)
    raise ValueError(f"未知的 KV 连接器: {config.kv_connector}")
