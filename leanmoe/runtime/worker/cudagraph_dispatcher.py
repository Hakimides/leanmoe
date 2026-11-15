"""CUDA Graph 模式与分桶调度。"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from leanmoe.config import CompilationConfig
from leanmoe.logger import get_logger, once

logger = get_logger("cudagraph")


class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int
    uniform_decode: bool
    # KV 长度桶；图内 gather buffer 的宽度
    len_bucket: int = 0

    @property
    def key(self) -> tuple[int, int, int, bool]:
        return (self.num_tokens, self.num_reqs, self.len_bucket, self.uniform_decode)


class CUDAGraphDispatcher:
    def __init__(
        self,
        config: CompilationConfig,
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        self.config = config
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.capture_sizes = config.resolved_capture_sizes(max_num_seqs)
        self.max_capture_size = max(self.capture_sizes) if self.capture_sizes else 0
        # KV 长度桶：不能只用一个上界，否则短序列也要按最长序列 gather。
        # 桶数直接决定图数量，按 vLLM 的"少桶 + 共享池"思路压到 3 个。
        self.len_buckets = [
            b for b in (256, 768, 2048) if b <= max_model_len
        ] or [max_model_len]
        self.mode = self._resolve_mode(config.level)

    @staticmethod
    def _resolve_mode(level: str) -> CUDAGraphMode:
        mapping = {
            "none": CUDAGraphMode.NONE,
            "piecewise": CUDAGraphMode.PIECEWISE,
            "full": CUDAGraphMode.FULL,
            "full_and_piecewise": CUDAGraphMode.FULL_AND_PIECEWISE,
        }
        return mapping[level]

    def _padded_size(self, num_tokens: int) -> int | None:
        for size in self.capture_sizes:
            if size >= num_tokens:
                return size
        return None

    def _len_bucket(self, max_seq_len: int) -> int | None:
        for b in self.len_buckets:
            if b >= max_seq_len:
                return b
        return None

    def dispatch(
        self,
        num_tokens: int,
        num_reqs: int,
        uniform_decode: bool,
        max_seq_len: int,
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None]:
        """返回本步实际使用的模式与分桶描述；None 表示走 eager。"""
        if self.mode == CUDAGraphMode.NONE:
            return CUDAGraphMode.NONE, None

        padded = self._padded_size(num_tokens)
        len_bucket = self._len_bucket(max_seq_len)
        if padded is None or len_bucket is None:
            if not self.config.allow_eager_fallback:
                raise RuntimeError(
                    f"num_tokens={num_tokens} / seq_len={max_seq_len} 超出捕获桶，"
                    "且配置禁止 eager 回退"
                )
            once.info(
                logger,
                "bucket_miss",
                f"num_tokens={num_tokens} seq_len={max_seq_len} 超出捕获桶，退回 eager",
            )
            return CUDAGraphMode.NONE, None

        # decode 用 FULL，prefill/mixed 用 PIECEWISE
        if uniform_decode and self.mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            # 图是按 token 桶捕获的，纯 decode 时 1 token/请求；
            # 请求数也补齐到桶，否则 key 里的 num_reqs 对不上，非 8 的倍数会掉 eager。
            bucket_reqs = padded if num_reqs == num_tokens else num_reqs
            return CUDAGraphMode.FULL, BatchDescriptor(
                padded, bucket_reqs, True, len_bucket
            )
        if self.mode in (
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            return CUDAGraphMode.PIECEWISE, BatchDescriptor(
                padded, num_reqs, False, len_bucket
            )
        return CUDAGraphMode.NONE, None
