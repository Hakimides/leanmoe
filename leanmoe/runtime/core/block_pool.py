"""块池：引用计数、空闲队列、块哈希表。"""

from __future__ import annotations

from typing import Iterable

from leanmoe.logger import get_logger

logger = get_logger("block_pool")


class KVCacheBlock:
    """KV 块：引用计数、哈希与空闲链表指针。"""
    __slots__ = ("block_id", "ref_cnt", "block_hash", "prev_free", "next_free")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: bytes | None = None
        self.prev_free: KVCacheBlock | None = None
        self.next_free: KVCacheBlock | None = None

    def __repr__(self) -> str:
        return (
            f"KVCacheBlock(id={self.block_id}, ref={self.ref_cnt}, "
            f"hash={'set' if self.block_hash else 'none'})"
        )


class FreeKVCacheBlockQueue:
    """双向链表实现的空闲块队列。"""

    def __init__(self, blocks: Iterable[KVCacheBlock]) -> None:
        self._head: KVCacheBlock | None = None
        self._tail: KVCacheBlock | None = None
        self.num_free_blocks = 0
        for block in blocks:
            self.append(block)

    def append(self, block: KVCacheBlock) -> None:
        """加到队尾。"""
        block.prev_free = self._tail
        block.next_free = None
        if self._tail is not None:
            self._tail.next_free = block
        else:
            self._head = block
        self._tail = block
        self.num_free_blocks += 1

    def prepend(self, block: KVCacheBlock) -> None:
        """加到队首。"""
        block.prev_free = None
        block.next_free = self._head
        if self._head is not None:
            self._head.prev_free = block
        else:
            self._tail = block
        self._head = block
        self.num_free_blocks += 1

    def popleft(self) -> KVCacheBlock | None:
        block = self._head
        if block is None:
            return None
        self._head = block.next_free
        if self._head is not None:
            self._head.prev_free = None
        else:
            self._tail = None
        block.prev_free = block.next_free = None
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n > self.num_free_blocks:
            raise ValueError(f"空闲块不足: 需要 {n}，剩余 {self.num_free_blocks}")
        blocks = []
        for _ in range(n):
            block = self.popleft()
            assert block is not None
            blocks.append(block)
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        """把块从队列中摘下（touch/命中时使用）。"""
        if block.prev_free is not None:
            block.prev_free.next_free = block.next_free
        else:
            self._head = block.next_free
        if block.next_free is not None:
            block.next_free.prev_free = block.prev_free
        else:
            self._tail = block.prev_free
        block.prev_free = block.next_free = None
        self.num_free_blocks -= 1

    def __len__(self) -> int:
        return self.num_free_blocks


class BlockPool:
    """块池：分配/回收块，并维护前缀哈希索引。"""
    def __init__(self, num_gpu_blocks: int, enable_caching: bool = True) -> None:
        if num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks 必须为正")
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)
        # block_hash -> block_id；只在 enable_caching 时使用
        self.cached_block_hash_to_block: dict[bytes, KVCacheBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.free_queue.num_free_blocks

    # ---------- 分配 ----------

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """取 n 个空闲块；带哈希的先从索引摘除。"""
        if num_blocks > self.num_free_blocks:
            raise ValueError(f"需要 {num_blocks} 块，只剩 {self.num_free_blocks}")
        blocks = self.free_queue.popleft_n(num_blocks)
        for block in blocks:
            assert block.ref_cnt == 0
            if block.block_hash is not None:
                self._evict_cached_block(block)
            block.ref_cnt = 1
        return blocks

    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """按哈希查找可复用块，引用计数 +1。"""
        if not self.enable_caching:
            return None
        block = self.cached_block_hash_to_block.get(block_hash)
        if block is None:
            return None
        # 命中：从空闲队列摘除并增加引用
        if block.ref_cnt == 0:
            self.free_queue.remove(block)
        block.ref_cnt += 1
        return block

    # ---------- 释放 ----------

    def free_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """引用计数归零后，按有无哈希回队首/队尾。"""
        for block in reversed(blocks):
            assert block.ref_cnt > 0
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.block_hash is None:
                    # 无哈希：队首，优先复用
                    self.free_queue.prepend(block)
                else:
                    # 有哈希：队尾，等待淘汰
                    self.free_queue.append(block)

    def _evict_cached_block(self, block: KVCacheBlock) -> None:
        if block.block_hash is not None:
            self.cached_block_hash_to_block.pop(block.block_hash, None)
            block.block_hash = None

    # ---------- 缓存登记 ----------

    def cache_full_blocks(
        self,
        blocks: list[KVCacheBlock],
        block_hashes: list[bytes],
        num_cached_blocks: int,
        num_full_blocks: int,
    ) -> None:
        """把已经写满的块登记进哈希表。"""
        if not self.enable_caching:
            return
        for i in range(num_cached_blocks, num_full_blocks):
            block = blocks[i]
            block_hash = block_hashes[i]
            if block_hash in self.cached_block_hash_to_block:
                continue
            block.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash] = block
            # 注意：这里的块仍被请求引用，不在空闲队列里；释放后会进入队尾
