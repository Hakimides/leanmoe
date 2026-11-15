"""KV cache：规格、哈希链、显存测算、分配管理器。"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

import torch

from leanmoe.logger import get_logger
from leanmoe.sequence import Request
from leanmoe.utils import cdiv
from leanmoe.runtime.core.block_pool import BlockPool, KVCacheBlock

logger = get_logger("kv_cache")


@dataclass
class KVCacheSpec:
    """单块 KV 的规格与字节数。"""
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def bytes_per_block(self) -> int:
        # K 和 V 各占一份
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * self.dtype.itemsize
        )


def hash_block_tokens(
    parent_hash: bytes | None,
    token_ids: list[int],
    extra_keys: tuple | None = None,
) -> bytes:
    """块哈希：parent + 本块 token + 额外键（LoRA / 多模态 / salt）。"""
    # 父哈希链保证只有完整前缀相同才命中
    payload = (
        parent_hash,
        tuple(token_ids),
        extra_keys or (),
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).digest()


def update_request_block_hashes(request: Request, block_size: int) -> None:
    """增量计算请求已满块的哈希，缓存在 request.block_hashes。"""
    token_ids = request.all_token_ids()
    num_full_blocks = len(token_ids) // block_size
    while len(request.block_hashes) < num_full_blocks:
        i = len(request.block_hashes)
        parent = request.block_hashes[-1] if request.block_hashes else None
        block_tokens = token_ids[i * block_size : (i + 1) * block_size]
        request.block_hashes.append(hash_block_tokens(parent, block_tokens))


@dataclass
class KVCacheConfig:
    num_blocks: int
    block_size: int
    spec: KVCacheSpec
    # 每层一块大 buffer 视图
    kv_cache_tensors: list[torch.Tensor]


def compute_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_size: int,
) -> int:
    """单个 block 的字节数（所有层）。"""
    return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size


def get_num_gpu_blocks(
    available_memory: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_size: int,
    override: int | None = None,
) -> int:
    """按可用显存计算 KV 块数。"""
    if override is not None:
        return override
    per_block = compute_kv_cache_bytes(
        num_layers, num_kv_heads, head_dim, block_size, dtype_size
    )
    return max(1, available_memory // per_block)


def allocate_kv_cache_tensors(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """分配一块大 buffer，并为每层返回 4D 视图。"""
    per_layer_elems = num_blocks * block_size * num_kv_heads * 2 * head_dim
    total_elems = per_layer_elems * num_layers
    buffer = torch.empty(total_elems, dtype=dtype, device=device)
    views = []
    for i in range(num_layers):
        start = i * per_layer_elems
        views.append(
            buffer[start : start + per_layer_elems].view(
                num_blocks, block_size, num_kv_heads, 2 * head_dim
            )
        )
    return views


class KVCacheManager:
    """块分配与 prefix cache 查询。"""

    def __init__(
        self,
        block_pool: BlockPool,
        block_size: int,
        enable_caching: bool = True,
    ) -> None:
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = enable_caching

    # ---------- 查询 ----------

    def get_computed_blocks(self, request: Request) -> tuple[list[KVCacheBlock], int]:
        """返回可复用的前缀块与该前缀的 token 数。"""
        if not self.enable_caching:
            return [], 0
        update_request_block_hashes(request, self.block_size)
        # 最后一个 token 必须重算（采样需要它的 logits），只复用完整块
        max_hit_blocks = (request.num_tokens - 1) // self.block_size
        computed: list[KVCacheBlock] = []
        for i in range(max_hit_blocks):
            block = self.block_pool.get_cached_block(request.block_hashes[i])
            if block is None:
                break
            computed.append(block)
        return computed, len(computed) * self.block_size

    # ---------- 分配 ----------

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: list[KVCacheBlock] | None = None,
        num_lookahead_tokens: int = 0,
    ) -> list[KVCacheBlock] | None:
        """尝试分配，空间不足返回 None（由调度器触发抢占）。"""
        new_computed_blocks = new_computed_blocks or []
        # 总需求必须基于"已计算 token 数 + 本步新增 + lookahead"，
        # 不能只看本步新增，否则跨 block 边界时会少分配。
        total_needed = (
            request.num_computed_tokens + num_new_tokens + num_lookahead_tokens
        )
        num_required_blocks = cdiv(total_needed, self.block_size)
        num_existing = len(request.block_ids) + len(new_computed_blocks)
        delta = num_required_blocks - num_existing
        if delta < 0:
            delta = 0
        if delta > self.block_pool.num_free_blocks:
            return None

        # 命中块加入请求的块表（引用计数已在 get_cached_block 中 +1）
        if new_computed_blocks:
            request.block_ids.extend(b.block_id for b in new_computed_blocks)

        new_blocks = self.block_pool.get_new_blocks(delta) if delta else []
        request.block_ids.extend(b.block_id for b in new_blocks)
        return new_blocks

    # ---------- 释放与登记 ----------

    def free(self, request: Request) -> None:
        """释放请求持有的全部块。"""
        if not request.block_ids:
            return
        blocks = [self.block_pool.blocks[i] for i in request.block_ids]
        self.block_pool.free_blocks(blocks)
        request.block_ids.clear()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """把本步已写满的块登记进哈希表。"""
        if not self.enable_caching:
            return
        update_request_block_hashes(request, self.block_size)
        num_full_blocks = min(
            num_computed_tokens // self.block_size,
            len(request.block_ids),
        )
        num_cached = request.num_cached_blocks
        if num_full_blocks <= num_cached:
            return
        blocks = [self.block_pool.blocks[i] for i in request.block_ids[:num_full_blocks]]
        self.block_pool.cache_full_blocks(
            blocks,
            request.block_hashes[:num_full_blocks],
            num_cached_blocks=num_cached,
            num_full_blocks=num_full_blocks,
        )
        request.num_cached_blocks = num_full_blocks

    # ---------- 统计 ----------

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free_blocks
